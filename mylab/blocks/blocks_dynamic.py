"""Blocks with memory: continuous states, discrete states and delays."""

from __future__ import annotations

import bisect
from collections import deque

import numpy as np

from .library import ONOFF, SAMPLE, BlockError, Node, P, block, sig

_RESET = ["none", "rising", "falling", "either", "level"]
_LIMITS = [ONOFF("LimitOutput", "off", "Limit the output"), P("UpperSaturationLimit", "inf", "Upper limit"),
           P("LowerSaturationLimit", "-inf", "Lower limit")]


def _fit(value: np.ndarray, shape) -> np.ndarray:
    """An initial value stretched to the shape of the block's input."""
    if shape is None or value.shape == tuple(shape):
        return value
    try:
        return np.array(np.broadcast_to(value, np.broadcast_shapes(value.shape, tuple(shape))), dtype=float)
    except ValueError:
        raise BlockError(f"the initial value has size {value.shape} but the input has size {tuple(shape)}") from None


def _edge(kind: str, prev, now) -> bool:
    if kind == "level":
        return bool(np.any(now != 0))
    if prev is None:
        return False
    up = bool(np.any((prev <= 0) & (now > 0)))
    down = bool(np.any((prev > 0) & (now <= 0)))
    return up if kind == "rising" else (down if kind == "falling" else up or down)


# =====================================================================================
# Continuous
# =====================================================================================

@block("Integrator", "Continuous", "Integrates its input over time. Optional limits, reset input and external initial value.",
       icon="∫")
class Integrator(Node):
    params = [P("InitialCondition", "0", "Initial value"),
              P("ExternalReset", "none", "Reset input", options=_RESET),
              P("InitialConditionSource", "internal", "Initial value from", options=["internal", "external"]),
              *_LIMITS, ONOFF("ShowStatePort", "off", "Show a second output with the state")]
    nin = ["sum", 1, ["if", "ExternalReset", "none", 0, 1], ["if", "InitialConditionSource", "external", 1, 0]]
    nout = ["sum", 1, ["if", "ShowStatePort", "on", 1, 0]]
    df = False
    sample = "cont"
    has_zc = True

    def init(self):
        self.reset = self.choice("ExternalReset")
        if self.reset not in _RESET:
            raise BlockError(f"unknown reset mode '{self.text('ExternalReset')}'")
        self.ext_ic = self.choice("InitialConditionSource") == "external"
        self.reset_port = 1 if self.reset != "none" else None
        self.ic_port = (2 if self.reset != "none" else 1) if self.ext_ic else None
        self.x0 = np.zeros(1) if self.ext_ic else self.num("InitialCondition")
        self.limited = self.is_on("LimitOutput")
        self.hi = self.num("UpperSaturationLimit") if self.limited else np.array([np.inf])
        self.lo = self.num("LowerSaturationLimit") if self.limited else np.array([-np.inf])
        if np.any(self.lo > self.hi):
            raise BlockError("the lower limit is above the upper limit")
        self.shape = self.x0.shape
        self.nx = self.x0.size
        self.sat = np.zeros(self.nx)
        self.r_prev = None
        self.state_port = self.is_on("ShowStatePort")
        self.has_zc = self.limited or self.reset != "none"

    def setup(self, shapes):
        want = shapes[0] if shapes else None
        if self.ic_port is not None and shapes and shapes[self.ic_port] is not None:
            want = np.broadcast_shapes(tuple(want or (1,)), tuple(shapes[self.ic_port]))
        x0 = _fit(self.x0, want)
        if self.limited:
            x0 = np.clip(x0, self.lo, self.hi)
        changed = x0.shape != self.shape or x0.size != self.nx
        self.x0, self.shape, self.nx = x0, x0.shape, x0.size
        if changed:
            self.sat = np.zeros(self.nx)
        return changed

    def late_init(self, u):
        if self.ic_port is not None:
            return np.ravel(np.broadcast_to(u[self.ic_port], self.shape), order="F")
        return None

    def initial_state(self):
        return np.ravel(self.x0, order="F")

    def _val(self):
        v = self.x.reshape(self.shape, order="F")
        return np.clip(v, self.lo, self.hi) if self.limited else v.copy()

    def out(self, t, u):
        v = self._val()
        return [v, v] if self.state_port else [v]

    def der(self, t, u):
        d = np.ravel(np.broadcast_to(u[0], self.shape), order="F").astype(float)
        if self.limited:
            d = np.where(self.sat != 0, 0.0, d)
        return d

    def update(self, t, u):
        if self.reset_port is not None:
            r = np.array(u[self.reset_port], copy=True)
            if _edge(self.reset, self.r_prev, r):
                ic = u[self.ic_port] if self.ic_port is not None else self.x0
                self.sim.set_state(self, np.ravel(np.broadcast_to(ic, self.shape), order="F"))
            self.r_prev = r
        if self.limited:
            x = self.x
            hi = np.ravel(np.broadcast_to(self.hi, self.shape), order="F")
            lo = np.ravel(np.broadcast_to(self.lo, self.shape), order="F")
            d = np.ravel(np.broadcast_to(u[0], self.shape), order="F")
            if np.any(x > hi) or np.any(x < lo):
                self.sim.set_state(self, np.clip(x, lo, hi))
                x = self.x
            self.sat = np.where((x >= hi) & (d > 0), 1.0, np.where((x <= lo) & (d < 0), -1.0, 0.0))

    def settle(self, t, u):
        """After states were reset in this step: decide again whether the limit holds the state."""
        if self.limited:
            x = self.x
            hi = np.ravel(np.broadcast_to(self.hi, self.shape), order="F")
            lo = np.ravel(np.broadcast_to(self.lo, self.shape), order="F")
            d = np.ravel(np.broadcast_to(u[0], self.shape), order="F")
            self.sat = np.where((x >= hi) & (d > 0), 1.0, np.where((x <= lo) & (d < 0), -1.0, 0.0))

    def zc(self, t, u):
        g = []
        if self.limited:
            x = self.x
            hi = np.ravel(np.broadcast_to(self.hi, self.shape), order="F")
            lo = np.ravel(np.broadcast_to(self.lo, self.shape), order="F")
            d = np.ravel(np.broadcast_to(u[0], self.shape), order="F")
            free = self.sat == 0
            g += list(np.where(free, x - hi, d)) + list(np.where(free, x - lo, d))
        if self.reset_port is not None:
            g += list(np.ravel(u[self.reset_port]))
        return g


class LTI(Node):
    """A linear system dx/dt = A x + B u, y = C x + D u."""

    sample = "cont"

    def set_system(self, a, b, c, d, x0=None):
        self.A, self.B, self.C, self.D = (np.atleast_2d(np.asarray(m, float)) for m in (a, b, c, d))
        n = self.A.shape[0] if self.A.size else 0
        if n == 0:
            self.A, self.B, self.C = np.zeros((0, 0)), np.zeros((0, self.D.shape[1])), np.zeros((self.D.shape[0], 0))
        if self.A.shape != (n, n) or self.B.shape[0] != n or self.C.shape[1] != n or \
                self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise BlockError(f"the matrices do not fit together: A is {self.A.shape}, B {self.B.shape}, C {self.C.shape}, "
                             f"D {self.D.shape}")
        self.n_states = n
        x0 = np.zeros(n) if x0 is None else np.ravel(x0)
        if x0.size == 1 and n != 1:
            x0 = np.full(n, float(x0[0]))
        if x0.size != n:
            raise BlockError(f"the initial state has {x0.size} values but the system has {n} states")
        self.x0 = x0.astype(float)
        self.df = bool(np.any(self.D != 0))

    def _u(self, u):
        v = np.ravel(u[0])
        if v.size != self.B.shape[1]:
            raise BlockError(f"the input has {v.size} elements but the block expects {self.B.shape[1]}")
        return v


class ContinuousLTI(LTI):
    def set_system(self, a, b, c, d, x0=None):
        super().set_system(a, b, c, d, x0)
        self.nx = self.n_states

    def initial_state(self):
        return self.x0

    def out(self, t, u):
        if not self.df:
            return [self.C @ self.x]
        return [self.C @ self.x + self.D @ self._u(u)]

    def der(self, t, u):
        return self.A @ self.x + self.B @ self._u(u)


def _tf(node: Node, num_name: str, den_name: str, inverse_powers: bool = False):
    from scipy import signal
    num, den = np.atleast_1d(node.num(num_name)), np.atleast_1d(node.num(den_name))
    if den.size == 0 or den[0] == 0 and not inverse_powers:
        den = np.trim_zeros(den, "f")
    if den.size == 0 or not np.any(den != 0):
        raise BlockError("the denominator is zero")
    if inverse_powers:  # polynomials in 1/z: pad to the same length, then they are polynomials in z
        n = max(num.size, den.size)
        num, den = np.pad(num, (0, n - num.size)), np.pad(den, (0, n - den.size))
        if den[0] == 0:
            raise BlockError("the first denominator coefficient must not be zero")
    num = np.trim_zeros(num, "f") if np.any(num != 0) else np.zeros(1)
    if num.size > den.size:
        raise BlockError("the numerator has a higher order than the denominator; the transfer function must be proper")
    if den.size == 1:
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[num[0] / den[0]]])
    return signal.tf2ss(num, den)


@block("TransferFcn", "Continuous", "Linear transfer function given by numerator and denominator coefficients in s.",
       title="Transfer Fcn", icon="tf")
class TransferFcn(ContinuousLTI):
    params = [P("Numerator", "[1]", "Numerator coefficients"), P("Denominator", "[1 1]", "Denominator coefficients")]

    def init(self):
        self.set_system(*_tf(self, "Numerator", "Denominator"))


@block("StateSpace", "Continuous", "Linear system dx/dt = Ax + Bu, y = Cx + Du.", title="State-Space", icon="ss")
class StateSpace(ContinuousLTI):
    params = [P("A", "1", "A"), P("B", "1", "B"), P("C", "1", "C"), P("D", "1", "D"),
              P("InitialCondition", "0", "Initial states")]

    def init(self):
        self.set_system(self.mat("A"), self.mat("B"), self.mat("C"), self.mat("D"), self.num("InitialCondition"))


@block("ZeroPole", "Continuous", "Linear transfer function given by its zeros, poles and gain.", title="Zero-Pole", icon="zpk")
class ZeroPole(ContinuousLTI):
    params = [P("Zeros", "[1]", "Zeros"), P("Poles", "[0 -1]", "Poles"), P("Gain", "1", "Gain")]

    def init(self):
        from scipy import signal
        z = np.ravel(np.asarray(self.value("Zeros"))) if self.text("Zeros") not in ("[]", "") else np.zeros(0)
        p = np.ravel(np.asarray(self.value("Poles")))
        if z.size > p.size:
            raise BlockError("there are more zeros than poles; the transfer function must be proper")
        num, den = signal.zpk2tf(z, p, self.scalar("Gain"))
        if np.iscomplexobj(num) or np.iscomplexobj(den):
            if np.max(np.abs(np.imag(num))) > 1e-9 or np.max(np.abs(np.imag(den))) > 1e-9:
                raise BlockError("complex zeros and poles must come in conjugate pairs")
        num, den = np.real(num), np.real(den)
        if den.size == 1:
            self.set_system(np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), [[float(np.ravel(num)[-1] / den[0])]])
        else:
            self.set_system(*signal.tf2ss(np.atleast_1d(num), den))


@block("PIDController", "Continuous", "PID controller with filtered derivative, output limits and anti-windup.",
       title="PID Controller", aliases=("PID", "PID1dof"), icon="PID")
class PIDController(Node):
    params = [P("Controller", "PID", "Controller", options=["PID", "PI", "PD", "P", "I"]),
              P("Form", "Parallel", "Form", options=["Parallel", "Ideal"]),
              P("P", "1", "Proportional (P)"), P("I", "1", "Integral (I)"), P("D", "0", "Derivative (D)"),
              P("N", "100", "Derivative filter coefficient (N)"),
              P("InitialConditionForIntegrator", "0", "Integrator initial value"),
              P("InitialConditionForFilter", "0", "Filter initial value"),
              *_LIMITS,
              P("AntiWindupMode", "none", "Anti-windup", options=["none", "back-calculation", "clamping"]),
              P("Kb", "1", "Back-calculation coefficient (Kb)")]
    sample = "cont"
    nx = 2

    def init(self):
        kind = self.text("Controller").upper()
        if kind not in ("PID", "PI", "PD", "P", "I"):
            raise BlockError(f"unknown controller type '{self.text('Controller')}'")
        self.kp = self.scalar("P") if "P" in kind else 0.0
        self.ki = self.scalar("I") if "I" in kind else 0.0
        self.kd = self.scalar("D") if "D" in kind else 0.0
        self.n = self.scalar("N") if self.kd != 0 else 1.0
        if self.kd != 0 and not np.isfinite(self.n):
            raise BlockError("the derivative filter coefficient N must be finite")
        self.ideal = self.choice("Form") == "ideal"
        if kind == "I":
            self.ideal = False
        self.limited = self.is_on("LimitOutput")
        self.hi = self.scalar("UpperSaturationLimit") if self.limited else np.inf
        self.lo = self.scalar("LowerSaturationLimit") if self.limited else -np.inf
        self.aw = self.choice("AntiWindupMode") if self.limited else "none"
        self.kb = self.scalar("Kb") if self.aw == "backcalculation" else 0.0
        self.x0 = np.array([self.scalar("InitialConditionForIntegrator"), self.scalar("InitialConditionForFilter")])

    def initial_state(self):
        return self.x0

    def _raw(self, e):
        d = self.n * (self.kd * e - self.x[1]) if self.kd != 0 else 0.0
        v = self.kp * (e + self.x[0] + d) if self.ideal else self.kp * e + self.x[0] + d
        return v, d

    def _e(self, u):
        if u[0].size != 1:
            raise BlockError("the PID Controller block takes a scalar input")
        return float(u[0].flat[0])

    def out(self, t, u):
        v, _ = self._raw(self._e(u))
        return [np.array([min(max(v, self.lo), self.hi)])]

    def der(self, t, u):
        e = self._e(u)
        v, d = self._raw(e)
        di = self.ki * e
        sat = min(max(v, self.lo), self.hi)
        if self.aw == "clamping" and v != sat and di * (v - sat) > 0:
            di = 0.0
        elif self.aw == "backcalculation":
            di += self.kb * (sat - v)
        return np.array([di, d])


@block("Derivative", "Continuous", "Approximate time derivative of the input (difference between solver steps).", icon="d/dt")
class Derivative(Node):
    params = []
    sample = "cont"

    def init(self):
        self.t_prev, self.u_prev, self.last = None, None, None

    def out(self, t, u):
        if self.t_prev is None or t <= self.t_prev:
            return [np.zeros_like(u[0]) if self.last is None else self.last + 0.0 * u[0]]
        return [(u[0] - self.u_prev) / (t - self.t_prev)]

    def update(self, t, u):
        if self.t_prev is not None and t > self.t_prev:
            self.last = (u[0] - self.u_prev) / (t - self.t_prev)
        if self.t_prev is None or t > self.t_prev:
            self.t_prev, self.u_prev = t, np.array(u[0], copy=True)


@block("TransportDelay", "Continuous", "Delays the input by a fixed time.", title="Transport Delay", icon="delay")
class TransportDelay(Node):
    params = [P("DelayTime", "1", "Time delay (s)"), P("InitialOutput", "0", "Output before the delay has passed")]
    sample = "cont"

    def init(self):
        self.delay = self.scalar("DelayTime")
        if self.delay < 0:
            raise BlockError("the time delay must not be negative")
        self.df = self.delay == 0
        self.y0 = self.num("InitialOutput")
        self.ts: list[float] = []
        self.us: list[np.ndarray] = []

    def setup(self, shapes):
        y0 = _fit(self.y0, shapes[0] if shapes else None)
        changed = y0.shape != self.y0.shape
        self.y0 = y0
        return changed

    def out(self, t, u):
        if self.delay == 0:
            return [u[0]]
        tq = t - self.delay
        ts = self.ts
        if not ts or tq < ts[0]:
            return [self.y0]
        if len(ts) == 1:
            return [self.us[0]]
        i = min(max(bisect.bisect_right(ts, tq) - 1, 0), len(ts) - 2)
        h = ts[i + 1] - ts[i]
        w = (tq - ts[i]) / h if h > 0 else 1.0
        return [self.us[i] + w * (self.us[i + 1] - self.us[i])]

    def update(self, t, u):
        if not self.ts or t > self.ts[-1]:
            self.ts.append(t)
            self.us.append(np.array(u[0], copy=True))
        else:
            self.us[-1] = np.array(u[0], copy=True)


# =====================================================================================
# Discrete
# =====================================================================================

class Held(Node):
    """A block that outputs a stored value and replaces it at the end of the step."""

    df = False
    ic_name = "InitialCondition"

    def init(self):
        self.state = self.num(self.ic_name)

    def setup(self, shapes):
        s = _fit(self.state, shapes[0] if shapes else None)
        changed = s.shape != self.state.shape
        self.state = s
        return changed

    def out(self, t, u):
        return [self.state]

    def update(self, t, u):
        self.state = np.array(u[0], dtype=float, copy=True)


@block("UnitDelay", "Discrete", "Delays the input by one sample time.", title="Unit Delay", icon="1/z")
class UnitDelay(Held):
    params = [P("InitialCondition", "0", "Initial value"), SAMPLE("-1")]
    sample = "param"


@block("Memory", "Discrete", "Outputs the input of the previous solver step.", icon="mem")
class Memory(Held):
    params = [P("InitialCondition", "0", "Initial value")]


@block("Delay", "Discrete", "Delays the input by a number of sample times.", icon="z⁻ⁿ")
class Delay(Node):
    params = [P("DelayLength", "2", "Number of samples"), P("InitialCondition", "0", "Initial value"), SAMPLE("-1")]
    sample = "param"
    df = False

    def init(self):
        self.n = int(self.scalar("DelayLength"))
        if self.n < 0:
            raise BlockError("the number of samples must not be negative")
        self.df = self.n == 0
        self.ic = self.num("InitialCondition")
        self.buf = deque([self.ic] * self.n)

    def setup(self, shapes):
        ic = _fit(self.ic, shapes[0] if shapes else None)
        changed = ic.shape != self.ic.shape
        self.ic = ic
        self.buf = deque([ic] * self.n)
        return changed

    def out(self, t, u):
        return [u[0] if self.n == 0 else self.buf[0]]

    def update(self, t, u):
        if self.n:
            self.buf.popleft()
            self.buf.append(np.array(u[0], dtype=float, copy=True))


@block("ZeroOrderHold", "Discrete", "Samples the input at its sample time and holds the value in between.",
       title="Zero-Order Hold", icon="zoh")
class ZeroOrderHold(Node):
    params = [SAMPLE("1")]
    sample = "param"

    def out(self, t, u):
        return [np.array(u[0], copy=True)]


@block("RateTransition", "Discrete", "Passes a signal between parts that run at different sample times (samples and holds).",
       title="Rate Transition", icon="rt")
class RateTransition(ZeroOrderHold):
    params = [P("OutPortSampleTime", "-1", "Output sample time (s); -1 takes it from the input")]

    def sample_time(self):
        return self.read_sample("OutPortSampleTime")


@block("DiscreteTimeIntegrator", "Discrete", "Integrates in steps of the sample time: forward Euler, backward Euler or trapezoidal.",
       title="Discrete-Time Integrator", aliases=("DiscreteIntegrator",), icon="KTs/(z−1)")
class DiscreteTimeIntegrator(Node):
    params = [P("IntegratorMethod", "Integration: Forward Euler", "Method",
                options=["Integration: Forward Euler", "Integration: Backward Euler", "Integration: Trapezoidal"]),
              P("gainval", "1.0", "Gain"), P("InitialCondition", "0", "Initial value"), *_LIMITS, SAMPLE("1")]
    sample = "param"

    def init(self):
        m = self.choice("IntegratorMethod")
        self.method = "backward" if "backward" in m else ("trapezoidal" if "trapez" in m else "forward")
        if "accumulation" in m:
            raise BlockError("the accumulation methods are not supported; use an integration method")
        self.df = self.method != "forward"
        self.k = self.num("gainval")
        self.state = self.num("InitialCondition")
        self.limited = self.is_on("LimitOutput")
        self.hi = self.num("UpperSaturationLimit") if self.limited else np.inf
        self.lo = self.num("LowerSaturationLimit") if self.limited else -np.inf

    def setup(self, shapes):
        s = _fit(self.state, shapes[0] if shapes else None)
        changed = s.shape != self.state.shape
        self.state = s
        return changed

    def _y(self, u):
        if self.method == "forward":
            y = self.state
        else:
            y = self.state + self.k * self.period * (0.5 if self.method == "trapezoidal" else 1.0) * u
        return np.clip(y, self.lo, self.hi) if self.limited else y

    def out(self, t, u):
        return [np.asarray(self._y(u[0]), float)]

    def update(self, t, u):
        step = self.k * self.period * u[0]
        if self.method == "forward":
            s = self.state + step
        elif self.method == "backward":
            s = self._y(u[0])
        else:
            s = self._y(u[0]) + 0.5 * step
        self.state = np.asarray(np.clip(s, self.lo, self.hi) if self.limited else s, float)


@block("DiscreteDerivative", "Discrete", "Difference between this sample and the last, divided by the sample time.",
       title="Discrete Derivative", icon="K(z−1)/Tsz")
class DiscreteDerivative(Node):
    params = [P("gainval", "1.0", "Gain"), P("ICPrevScaledInput", "0", "Initial value of the previous input"), SAMPLE("-1")]
    sample = "param"

    def init(self):
        self.k = self.num("gainval")
        self.prev = self.num("ICPrevScaledInput")

    def out(self, t, u):
        return [np.asarray(self.k * (u[0] - self.prev) / self.period, float)]

    def update(self, t, u):
        self.prev = np.array(u[0], dtype=float, copy=True)


class DiscreteLTI(LTI):
    sample = "param"

    def set_system(self, a, b, c, d, x0=None):
        super().set_system(a, b, c, d, x0)
        self.state = self.x0.copy()

    def out(self, t, u):
        if not self.df:
            return [self.C @ self.state]
        return [self.C @ self.state + self.D @ self._u(u)]

    def update(self, t, u):
        self.state = self.A @ self.state + self.B @ self._u(u)


@block("DiscreteTransferFcn", "Discrete", "Discrete transfer function given by numerator and denominator coefficients in z.",
       title="Discrete Transfer Fcn", icon="tf(z)")
class DiscreteTransferFcn(DiscreteLTI):
    params = [P("Numerator", "[1]", "Numerator coefficients"), P("Denominator", "[1 0.5]", "Denominator coefficients"), SAMPLE("1")]

    def init(self):
        self.set_system(*_tf(self, "Numerator", "Denominator"))


@block("DiscreteFilter", "Discrete", "Discrete filter given by numerator and denominator coefficients in 1/z.",
       title="Discrete Filter", icon="filt")
class DiscreteFilter(DiscreteLTI):
    params = [P("Numerator", "[1]", "Numerator coefficients"), P("Denominator", "[1 0.5]", "Denominator coefficients"), SAMPLE("1")]

    def init(self):
        self.set_system(*_tf(self, "Numerator", "Denominator", inverse_powers=True))


@block("DiscreteFir", "Discrete", "Discrete FIR filter: a weighted sum of the latest input samples.", title="Discrete FIR Filter",
       icon="fir")
class DiscreteFir(DiscreteLTI):
    params = [P("Coefficients", "[0.5 0.5]", "Coefficients"), SAMPLE("1")]

    def init(self):
        self.raw["Denominator"] = "1"
        self.raw["Numerator"] = self.raw["Coefficients"]
        self.set_system(*_tf(self, "Numerator", "Denominator", inverse_powers=True))


@block("DiscreteStateSpace", "Discrete", "Discrete linear system x[k+1] = Ax[k] + Bu[k], y[k] = Cx[k] + Du[k].",
       title="Discrete State-Space", icon="ss(z)")
class DiscreteStateSpace(DiscreteLTI):
    params = [P("A", "1", "A"), P("B", "1", "B"), P("C", "1", "C"), P("D", "1", "D"),
              P("InitialCondition", "0", "Initial states"), SAMPLE("1")]

    def init(self):
        self.set_system(self.mat("A"), self.mat("B"), self.mat("C"), self.mat("D"), self.num("InitialCondition"))


__all__ = ["sig"]
