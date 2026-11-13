"""Sources, sinks, math, logic and static nonlinear blocks."""

from __future__ import annotations

import math

import numpy as np

from .library import INF, ONOFF, SAMPLE, BlockError, Node, P, block, elementwise, sig

TOL = 1e-9


def _after(t: float, when: np.ndarray) -> np.ndarray:
    """True where time ``t`` has reached ``when`` (with a tolerance for rounding)."""
    return t >= when - TOL * np.maximum(1.0, np.abs(when))


def _next(t: float, times) -> float:
    """Smallest of ``times`` that lies after ``t``."""
    best = INF
    for v in np.ravel(times):
        if v > t + TOL * max(1.0, abs(v)) and v < best:
            best = float(v)
    return best


# =====================================================================================
# Sources
# =====================================================================================


class Source(Node):
    nin = 0
    sample = "param"

    def sample_time(self):
        ts = self.read_sample("SampleTime", None) if "SampleTime" in self.raw else None
        return ts


@block("Constant", "Sources", "Outputs a constant value (a number, vector or matrix).", icon="1")
class Constant(Source):
    params = [P("Value", "1", "Constant value"), ONOFF("VectorParams1D", "on", "Treat a row or column as a 1-D vector")]
    sample = "const"

    def sample_time(self):
        return "const"

    def init(self):
        v = self.mat("Value")
        self.v = sig(v) if self.is_on("VectorParams1D") or v.size == 1 else v

    def out(self, t, u):
        return [self.v]


@block("Step", "Sources", "Jumps from one value to another at a given time.", icon="step")
class Step(Source):
    params = [
        P("Time", "1", "Step time"),
        P("Before", "0", "Value before the step"),
        P("After", "1", "Value after the step"),
        SAMPLE("0"),
    ]

    def init(self):
        self.time, self.before, self.after = self.num("Time"), self.num("Before"), self.num("After")
        self.on = _after(self.ctx.t0, self.time)

    def out(self, t, u):
        if not self.sim.minor:
            self.on = _after(t, self.time)
        return [np.where(self.on, self.after, self.before).astype(float) + 0.0 * self.time]

    def next_break(self, t):
        return _next(t, self.time)


@block("Ramp", "Sources", "Rises (or falls) at a constant slope from a start time on.", icon="ramp")
class Ramp(Source):
    params = [P("Slope", "1", "Slope"), P("Start", "0", "Start time"), P("InitialOutput", "0", "Initial output")]
    sample = "cont"

    def init(self):
        self.slope, self.start, self.y0 = self.num("Slope"), self.num("Start"), self.num("InitialOutput")

    def out(self, t, u):
        return [self.y0 + self.slope * np.maximum(t - self.start, 0.0)]

    def next_break(self, t):
        return _next(t, self.start)


@block(
    "SineWave",
    "Sources",
    "Sine wave: Amplitude*sin(Frequency*t + Phase) + Bias, frequency in rad/s.",
    aliases=("Sin",),
    icon="sine",
)
class SineWave(Source):
    params = [
        P("Amplitude", "1", "Amplitude"),
        P("Bias", "0", "Bias"),
        P("Frequency", "1", "Frequency (rad/s)"),
        P("Phase", "0", "Phase (rad)"),
        P("TimeSource", "Use simulation time", "Time", options=["Use simulation time", "Use external signal"]),
        SAMPLE("0"),
    ]
    nin = ["if", "TimeSource", "Use external signal", 1, 0]

    def init(self):
        self.a, self.b, self.f, self.ph = (
            self.num("Amplitude"),
            self.num("Bias"),
            self.num("Frequency"),
            self.num("Phase"),
        )

    def out(self, t, u):
        if u:  # time comes from the input
            t = u[0]
        return [self.a * np.sin(self.f * t + self.ph) + self.b]


@block(
    "PulseGenerator",
    "Sources",
    "Rectangular pulses with a given period, width (percent of the period) and delay.",
    aliases=("DiscretePulseGenerator",),
    icon="pulse",
)
class PulseGenerator(Source):
    params = [
        P("Amplitude", "1", "Amplitude"),
        P("Period", "2", "Period (s)"),
        P("PulseWidth", "50", "Pulse width (% of period)"),
        P("PhaseDelay", "0", "Delay before the first pulse (s)"),
        SAMPLE("0"),
    ]

    def init(self):
        self.amp, self.per = self.num("Amplitude"), self.scalar("Period")
        self.width, self.delay = self.scalar("PulseWidth") / 100.0 * self.per, self.scalar("PhaseDelay")
        if self.per <= 0 or not 0 <= self.width <= self.per:
            raise BlockError("the period must be positive and the pulse width between 0 and 100 percent")
        self.on = self._on(self.ctx.t0)

    def _on(self, t):
        if t < self.delay - TOL:
            return False
        k = math.floor((t - self.delay) / self.per + TOL)
        return (t - self.delay - k * self.per) < self.width - TOL * self.per

    def out(self, t, u):
        if not self.sim.minor:
            self.on = self._on(t)
        return [self.amp * (1.0 if self.on else 0.0)]

    def next_break(self, t):
        k = max(math.floor((t - self.delay) / self.per) - 1, 0)
        return _next(t, [self.delay + (k + i) * self.per + w for i in range(4) for w in (0.0, self.width)])


@block(
    "SignalGenerator",
    "Sources",
    "Sine, square, sawtooth or random wave with a given amplitude and frequency.",
    icon="wave",
)
class SignalGenerator(Source):
    params = [
        P("WaveForm", "sine", "Wave form", options=["sine", "square", "sawtooth", "random"]),
        P("Amplitude", "1", "Amplitude"),
        P("Frequency", "1", "Frequency"),
        P("Units", "Hertz", "Frequency units", options=["Hertz", "rad/sec"]),
    ]
    sample = "cont"

    def init(self):
        self.form, self.amp = self.choice("WaveForm"), self.num("Amplitude")
        self.hz = self.scalar("Frequency") / (1.0 if self.choice("Units") == "hertz" else 2 * math.pi)
        if self.form not in ("sine", "square", "sawtooth", "random"):
            raise BlockError(f"unknown wave form '{self.text('WaveForm')}'")
        self.k = math.floor(2 * self.hz * self.ctx.t0 + TOL)
        self.rng = np.random.RandomState(0)
        self.rnd = np.zeros_like(self.amp)

    def out(self, t, u):
        if not self.sim.minor:
            self.k = math.floor(2 * self.hz * t + TOL)  # number of half periods so far
            if self.form == "random":
                self.rnd = self.amp * self.rng.uniform(-1, 1, self.amp.shape)
        if self.form == "sine":
            return [self.amp * math.sin(2 * math.pi * self.hz * t)]
        if self.form == "square":
            return [self.amp * (1.0 if self.k % 2 == 0 else -1.0)]
        if self.form == "sawtooth":
            return [self.amp * (2 * (self.hz * t - self.k // 2) - 1)]
        return [self.rnd]

    def next_break(self, t):
        if self.form in ("sine", "random") or self.hz <= 0:
            return INF
        half = 0.5 / self.hz
        k = math.floor(t / half)
        return _next(t, [(k + i) * half for i in range(3) if self.form == "square" or (k + i) % 2 == 0])


@block(
    "Chirp",
    "Sources",
    "Sine wave whose frequency rises linearly from f1 to f2 (Hz) over the target time.",
    title="Chirp Signal",
    icon="chirp",
)
class Chirp(Source):
    params = [
        P("f1", "0.1", "Initial frequency (Hz)"),
        P("T", "100", "Target time (s)"),
        P("f2", "1", "Frequency at target time (Hz)"),
    ]
    sample = "cont"

    def init(self):
        self.f1, self.T, self.f2 = self.scalar("f1"), self.scalar("T"), self.scalar("f2")

    def out(self, t, u):
        return [np.array([math.cos(2 * math.pi * (self.f1 * t + (self.f2 - self.f1) * t * t / (2 * self.T)))])]


@block("Clock", "Sources", "Outputs the simulation time.", aliases=("DigitalClock",), icon="clock")
class Clock(Source):
    params = []
    sample = "cont"

    def out(self, t, u):
        return [np.array([t])]


@block("Ground", "Sources", "Outputs zero.", icon="0")
class Ground(Source):
    params = []
    sample = "const"

    def sample_time(self):
        return "const"

    def out(self, t, u):
        return [np.zeros(1)]


class _Random(Source):
    def sample_time(self):
        return self.read_sample("SampleTime", None)

    def init(self):
        seeds = self.num("Seed")  # a vector of seeds gives a vector signal, one stream per element
        self.seed_shape = np.zeros(seeds.shape if seeds.size > 1 else (1,))
        self.rng = np.random.RandomState([int(v) % (2**32) for v in np.ravel(seeds)])
        self.prepare()
        self.val = self.draw() * 0.0

    def out(self, t, u):
        if not self.sim.minor:
            self.val = self.draw()
        return [self.val]


@block("RandomNumber", "Sources", "Normally distributed random numbers, a new one every sample time.", icon="rand")
class RandomNumber(_Random):
    params = [P("Mean", "0", "Mean"), P("Variance", "1", "Variance"), P("Seed", "0", "Seed"), SAMPLE("0.1")]

    def prepare(self):
        self.mean, self.sd = self.num("Mean"), np.sqrt(self.num("Variance"))

    def draw(self):
        shape = np.broadcast(self.mean, self.sd, self.seed_shape).shape
        return self.mean + self.sd * self.rng.standard_normal(shape)


@block(
    "UniformRandomNumber",
    "Sources",
    "Uniformly distributed random numbers between a minimum and a maximum.",
    icon="rand",
)
class UniformRandomNumber(_Random):
    params = [P("Minimum", "-1", "Minimum"), P("Maximum", "1", "Maximum"), P("Seed", "0", "Seed"), SAMPLE("0.1")]

    def prepare(self):
        self.lo, self.hi = self.num("Minimum"), self.num("Maximum")

    def draw(self):
        shape = np.broadcast(self.lo, self.hi, self.seed_shape).shape
        return self.lo + (self.hi - self.lo) * self.rng.uniform(size=shape)


@block(
    "BandLimitedWhiteNoise",
    "Sources",
    "White noise for continuous systems: random values of variance Cov/Ts held for Ts.",
    title="Band-Limited White Noise",
    icon="noise",
)
class BandLimitedWhiteNoise(_Random):
    params = [P("Cov", "0.1", "Noise power"), P("Ts", "0.1", "Sample time (s)"), P("seed", "23341", "Seed")]

    def sample_time(self):
        return self.read_sample("Ts", None)

    def init(self):
        self.raw["Seed"] = self.raw.get("seed", "0")
        super().init()

    def prepare(self):
        self.cov, self.ts = self.num("Cov"), self.scalar("Ts")

    def draw(self):
        shape = np.broadcast(self.cov, self.seed_shape).shape
        return np.sqrt(self.cov / self.ts) * self.rng.standard_normal(shape)


def _table(node: Node, v) -> tuple[np.ndarray, np.ndarray]:
    """Times and values from a matrix [t, u...] or a struct with time and data."""
    from ..values import Struct

    if isinstance(v, Struct):
        f = {name.lower(): name for name in v.fields}
        if "time" in f and "data" in f:
            t, d = v.get_field(0, f["time"]), v.get_field(0, f["data"])
        elif "time" in f and "signals" in f:
            t, s = v.get_field(0, f["time"]), v.get_field(0, f["signals"])
            d = s.get_field(0, "values")
        else:
            raise BlockError("the structure needs the fields Time and Data (or time and signals.values)")
        t, d = np.asarray(t, float).reshape(-1), np.asarray(d, float)
        d = d.reshape(t.size, -1) if d.size else d
    else:
        m = np.asarray(v, float)
        if m.ndim != 2 or m.shape[1] < 2:
            raise BlockError("the data must be a matrix with time in the first column and values in the others")
        t, d = m[:, 0], m[:, 1:]
    if t.size == 0 or d.shape[0] != t.size or np.any(np.diff(t) < 0):
        raise BlockError("the time values must be increasing and match the number of data rows")
    return t, d


@block(
    "FromWorkspace",
    "Sources",
    "Plays back data from a workspace variable: a matrix [t, u] or a struct with Time and Data.",
    icon="ws",
)
class FromWorkspace(Source):
    params = [
        P("VariableName", "simin", "Data (variable or expression)"),
        ONOFF("Interpolate", "on", "Interpolate between the data points"),
        P(
            "OutputAfterFinalValue",
            "Extrapolation",
            "After the last data point",
            options=["Extrapolation", "Setting to zero", "Holding final value"],
        ),
        SAMPLE("0"),
    ]

    def init(self):
        self.t, self.d = _table(self, self.value("VariableName"))
        self.after = self.choice("OutputAfterFinalValue")
        self.interp = self.is_on("Interpolate")

    def out(self, t, u):
        ts, d = self.t, self.d
        if t > ts[-1] and self.after != "extrapolation":
            return [d[-1] * (0.0 if self.after == "settingtozero" else 1.0)]
        if ts.size == 1:
            return [d[0].copy()]
        i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, ts.size - 2))
        if not self.interp:
            return [d[i + 1 if t >= ts[i + 1] else i].copy()]
        h = ts[i + 1] - ts[i]
        w = (t - ts[i]) / h if h > 0 else 1.0
        return [d[i] + w * (d[i + 1] - d[i])]

    def next_break(self, t):
        return _next(t, self.t) if self.t.size <= 2000 else INF


@block(
    "RepeatingSequence",
    "Sources",
    "Repeats a piecewise-linear wave given by time values and output values.",
    icon="saw",
)
class RepeatingSequence(Source):
    params = [P("rep_seq_t", "[0 2]", "Time values"), P("rep_seq_y", "[0 2]", "Output values")]
    sample = "cont"

    def init(self):
        self.tv, self.yv = self.num("rep_seq_t"), self.num("rep_seq_y")
        if (
            self.tv.size != self.yv.size
            or self.tv.size < 2
            or np.any(np.diff(self.tv) < 0)
            or self.tv[-1] <= self.tv[0]
        ):
            raise BlockError("time values and output values must have the same length, and time must increase")
        self.per = self.tv[-1] - self.tv[0]
        self.k = math.floor((self.ctx.t0 - self.tv[0]) / self.per + TOL)

    def out(self, t, u):
        if not self.sim.minor:
            self.k = math.floor((t - self.tv[0]) / self.per + TOL)
        tau = min(max(t - self.k * self.per, self.tv[0]), self.tv[-1])
        return [np.array([np.interp(tau, self.tv, self.yv)])]

    def next_break(self, t):
        k = math.floor((t - self.tv[0]) / self.per)
        return _next(t, [self.tv + (k + i) * self.per for i in (0, 1)])


@block(
    "RepeatingSequenceStair",
    "Sources",
    "Steps through a list of values, one per sample time, and starts again.",
    icon="stair",
)
class RepeatingSequenceStair(Source):
    params = [P("OutValues", "[3 1 4 2 1]", "Output values"), P("tsamp", "1", "Sample time (s)")]

    def sample_time(self):
        return self.read_sample("tsamp", None)

    def init(self):
        self.v = self.num("OutValues")
        self.ts = self.scalar("tsamp")

    def out(self, t, u):
        return [self.v[[int(math.floor(t / self.ts + TOL)) % self.v.size]]]


@block(
    "Inport",
    "Routing",
    "Input of a subsystem. At the top level: an external input (zero unless one is given to sim).",
    title="In",
    aliases=("In1",),
    icon="in",
)
class Inport(Source):
    params = [P("Port", "1", "Port number", "text")]
    sample = "cont"

    def init(self):
        self.port = int(float(self.text("Port") or 1))

    def out(self, t, u):
        return [self.ctx.external_input(self.port, t)]


# =====================================================================================
# Sinks
# =====================================================================================


class Sink(Node):
    nout = 0
    is_sink = True
    decimation = 1

    def init(self):
        self.log_t: list[float] = []
        self.log_u: list[list] = [[] for _ in range(self.n_in())]
        self.count = 0
        self.last_count = -2

    def out(self, t, u):
        if self.sim.logging:
            if self.log_t and self.log_t[-1] == t and self.last_count == self.count - 1:
                for k, v in enumerate(u):  # same instant again (after a state reset): keep the newer values
                    self.log_u[k][-1] = np.ravel(v, order="F").copy()
                return []
            if self.count % self.decimation == 0:
                self.log_t.append(t)
                self.last_count = self.count
                for k, v in enumerate(u):
                    self.log_u[k].append(np.ravel(v, order="F").copy())
            self.count += 1
        return []

    def data(self, port: int = 0) -> np.ndarray:
        rows = self.log_u[port]
        if not rows:
            return np.zeros((0, 1))
        width = max(r.size for r in rows)
        return np.array([np.broadcast_to(r, (width,)) if r.size in (1, width) else np.resize(r, width) for r in rows])


@block(
    "Scope",
    "Sinks",
    "Plots its inputs against time. Each input port gets its own axes.",
    aliases=("Record",),
    icon="scope",
)
class Scope(Sink):
    params = [P("NumInputPorts", "1", "Number of input ports", "text"), P("Title", "", "Title", "text")]
    nin = ["int", "NumInputPorts"]


@block("ToWorkspace", "Sinks", "Saves its input to a variable (and a field of the object returned by sim).", icon="ws")
class ToWorkspace(Sink):
    params = [
        P("VariableName", "simout", "Variable name", "text"),
        P("SaveFormat", "Timeseries", "Format", options=["Timeseries", "Array", "Structure With Time", "Structure"]),
        P("Decimation", "1", "Keep every n-th sample"),
        SAMPLE("-1"),
    ]
    sample = "param"

    def init(self):
        super().init()
        self.decimation = max(int(self.scalar("Decimation")), 1)
        self.var = self.text("VariableName")
        if not self.var.isidentifier():
            raise BlockError(f"'{self.var}' is not a valid variable name")
        self.format = self.choice("SaveFormat")


@block("Display", "Sinks", "Shows the current value of its input.", icon="disp")
class Display(Sink):
    params = []

    def out(self, t, u):
        if self.sim.logging:
            self.last = np.array(u[0], copy=True)
        return []


@block("XYGraph", "Sinks", "Plots the second input against the first.", title="XY Graph", icon="xy")
class XYGraph(Sink):
    params = []
    nin = 2


@block("Terminator", "Sinks", "Ends a signal that is not used.", icon="term")
class Terminator(Node):
    params = []
    nout = 0
    is_sink = True


@block("Viewer", "Sinks", "Stands for a display that mylab does not draw (video, 3-D animation); its inputs are ignored.", icon="view")
class Viewer(Node):
    params = [P("NumInputPorts", "1", "Number of input ports", "text"), P("Kind", "", "What it showed", "text")]
    nin = ["int", "NumInputPorts"]
    nout = 0
    is_sink = True


@block("StopSimulation", "Sinks", "Stops the simulation when its input is not zero.", aliases=("Stop",), icon="stop")
class StopSimulation(Node):
    params = []
    nout = 0
    is_sink = True

    def out(self, t, u):
        if not self.sim.minor and np.any(u[0] != 0):
            self.sim.request_stop(self.path)
        return []


@block(
    "Outport",
    "Routing",
    "Output of a subsystem. At the top level its signal is logged in yout.",
    title="Out",
    aliases=("Out1",),
    icon="out",
)
class Outport(Sink):
    params = [P("Port", "1", "Port number", "text"), P("InitialOutput", "[]", "Output before the subsystem first runs")]


# =====================================================================================
# Math
# =====================================================================================


def _transpose(u: np.ndarray) -> np.ndarray:
    """Transpose; a vector (which counts as a column) becomes a 1-by-n row matrix."""
    if u.ndim == 1:
        return u.reshape(1, -1).astype(float) if u.size > 1 else u
    return sig(u.T)


def _matmul(a: np.ndarray, b: np.ndarray, what: str) -> np.ndarray:
    try:
        if a.size == 1 or b.size == 1:
            return a * b
        if a.ndim == 1 and b.ndim == 2 and b.shape[0] == 1:
            return a.reshape(-1, 1) @ b  # a vector counts as a column: column times row is a full matrix
        if a.ndim == 2 and b.ndim == 1 and a.shape[1] != b.size and a.shape[1] == 1:
            return a @ b.reshape(1, -1)
        return a @ b
    except ValueError:
        raise BlockError(f"{what}: sizes {a.shape} and {b.shape} cannot be multiplied") from None


@block("Gain", "Math", "Multiplies its input by a constant: element by element, or as a matrix product.", icon="K")
class Gain(Node):
    params = [
        P("Gain", "1", "Gain"),
        P(
            "Multiplication",
            "Element-wise(K.*u)",
            "Multiplication",
            options=["Element-wise(K.*u)", "Matrix(K*u)", "Matrix(u*K)", "Matrix(K*u) (u vector)"],
        ),
    ]

    def init(self):
        self.mode = self.choice("Multiplication")
        self.k = self.num("Gain") if self.mode.startswith("element") else self.mat("Gain")
        self.same_shape = self.mode.startswith("element") and self.k.size == 1

    def want_in(self, port):
        if self.mode.startswith("element") or self.k.ndim != 2 or self.k.size == 1:
            return None
        n = self.k.shape[0] if self.mode == "matrixuk" else self.k.shape[1]
        return (n,) if n > 1 else None

    def out(self, t, u):
        if self.mode.startswith("element"):
            try:
                return [self.k * u[0]]
            except ValueError:
                raise BlockError(f"gain of size {self.k.shape} does not fit an input of size {u[0].shape}") from None
        if self.mode == "matrixuk":
            return [sig(_matmul(u[0], self.k, "u*K"))]
        return [sig(_matmul(self.k, u[0], "K*u"))]


def _add(vals: list, signs: str) -> np.ndarray:
    try:
        total = vals[0] if signs[0] == "+" else -vals[0]
        for v, s in zip(vals[1:], signs[1:]):
            total = total + v if s == "+" else total - v
        return total
    except ValueError:
        raise BlockError(
            "the inputs have sizes that cannot be added: " + ", ".join(str(v.shape) for v in vals)
        ) from None


@block(
    "Sum",
    "Math",
    "Adds and subtracts its inputs; the signs string (such as +-) gives one sign per input.",
    aliases=("Add", "Subtract"),
    icon="Σ",
)
class Sum(Node):
    same_shape = True

    params = [P("Inputs", "++", "List of signs (or number of inputs)", "text")]
    nin = ["signs", "Inputs"]

    def init(self):
        s = self.text("Inputs").replace("|", "").replace(" ", "")
        self.signs = "+" * int(s) if s.isdigit() else s
        if not self.signs or set(self.signs) - set("+-"):
            raise BlockError(f"the list of signs '{self.text('Inputs')}' may only contain + and -")

    def out(self, t, u):
        if len(u) == 1:
            return [np.array([np.sum(u[0])]) * (1.0 if self.signs[0] == "+" else -1.0)]
        return [_add(u, self.signs)]


@block(
    "Product",
    "Math",
    "Multiplies and divides its inputs; the string (such as */) gives one operation per input.",
    icon="×",
)
class Product(Node):
    params = [
        P("Inputs", "2", "Number of inputs, or a list of * and /", "text"),
        P("Multiplication", "Element-wise(.*)", "Multiplication", options=["Element-wise(.*)", "Matrix(*)"]),
    ]
    nin = ["signs", "Inputs"]

    def init(self):
        s = self.text("Inputs").replace("|", "").replace(" ", "")
        self.ops = "*" * int(s) if s.isdigit() else s
        if not self.ops or set(self.ops) - set("*/"):
            raise BlockError(f"the list '{self.text('Inputs')}' may only contain * and /")
        self.matrix = self.choice("Multiplication").startswith("matrix")

    def out(self, t, u):
        with np.errstate(divide="ignore", invalid="ignore"):
            if len(u) == 1 and not self.matrix:
                p = np.array([np.prod(u[0])])
                return [p if self.ops[0] == "*" else 1.0 / p]
            if self.matrix:
                total = None
                for v, op in zip(u, self.ops):
                    if op == "/":
                        v = np.linalg.inv(np.atleast_2d(v)) if v.size > 1 else 1.0 / v
                    total = v if total is None else _matmul(total, v, "matrix product")
                if total.ndim == 2 and total.shape[0] == 1 and total.size > 1:
                    return [total.astype(float)]  # row times matrix stays a row
                return [sig(total)]
            try:
                total = u[0] if self.ops[0] == "*" else 1.0 / u[0]
                for v, op in zip(u[1:], self.ops[1:]):
                    total = total * v if op == "*" else total / v
            except ValueError:
                raise BlockError(
                    "the inputs have sizes that cannot be multiplied: " + ", ".join(str(v.shape) for v in u)
                ) from None
            return [np.asarray(total, float)]


@block("Divide", "Math", "Divides the first input by the second.", icon="÷")
class Divide(Product):
    params = [P("Inputs", "*/", "List of * and /", "text"), Product.params[1]]


@block("Abs", "Math", "Absolute value.", icon="|u|")
class Abs(Node):
    same_shape = True

    params = []
    has_zc = True

    def out(self, t, u):
        v = u[0]
        return [np.where(self.latched(lambda: v >= 0), v, -v)]

    def zc(self, t, u):
        return list(np.ravel(u[0]))


@block("Sign", "Math", "Sign of the input: -1, 0 or 1.", aliases=("Signum",), icon="sgn")
class Sign(Node):
    params = []
    has_zc = True

    def out(self, t, u):
        v = u[0]
        return [self.latched(lambda: np.sign(v)) + 0.0 * v]

    def zc(self, t, u):
        return list(np.ravel(u[0]))


elementwise("UnaryMinus", "Math", "Negates the input.", lambda u, b: -u, icon="−u")
elementwise(
    "Bias",
    "Math",
    "Adds a constant to the input.",
    lambda u, b: u + b.num("Bias"),
    [P("Bias", "0", "Bias")],
    icon="u+b",
)
elementwise(
    "SumOfElements",
    "Math",
    "Adds up the elements of a vector.",
    lambda u, b: np.array([np.sum(u)]),
    title="Sum of Elements",
    icon="Σu",
)
elementwise(
    "ProductOfElements",
    "Math",
    "Multiplies the elements of a vector.",
    lambda u, b: np.array([np.prod(u)]),
    title="Product of Elements",
    icon="Πu",
)
elementwise(
    "Polynomial",
    "Math",
    "Evaluates a polynomial with the given coefficients (highest power first).",
    lambda u, b: np.polyval(b.num("Coefs"), u),
    [P("Coefs", "[1 0]", "Coefficients")],
    aliases=("Polyval",),
    icon="P(u)",
)
elementwise(
    "Width", "Math", "Outputs the number of elements of its input.", lambda u, b: np.array([float(u.size)]), icon="#u"
)
elementwise(
    "DataTypeConversion",
    "Math",
    "Passes the signal on (all signals are double precision in mylab Blocks).",
    lambda u, b: u,
    title="Data Type Conversion",
    aliases=("SignalConversion", "SignalSpecification", "InitialCondition"),
    icon="→",
)

_MATH2 = ["pow", "hypot", "rem", "mod"]
_MATH = {
    "exp": np.exp,
    "log": np.log,
    "10^u": lambda u: 10.0**u,
    "log10": np.log10,
    "magnitude^2": lambda u: u * u,
    "square": lambda u: u * u,
    "reciprocal": lambda u: 1.0 / u,
    "conj": lambda u: u,
    "transpose": lambda u: _transpose(u),
    "hermitian": lambda u: _transpose(u),
    "sqrt": np.sqrt,
    "pow": np.power,
    "hypot": np.hypot,
    "rem": np.fmod,
    "mod": np.mod,
}


@block(
    "MathFunction",
    "Math",
    "A choice of functions: exp, log, 10^u, log10, square, pow, reciprocal, hypot, rem, mod, transpose.",
    aliases=("Math",),
    icon="f(u)",
)
class MathFunction(Node):
    params = [P("Operator", "exp", "Function", options=list(_MATH))]
    nin = ["if", "Operator", _MATH2, 2, 1]

    def init(self):
        self.fn = _MATH.get(self.text("Operator"))
        if self.fn is None:
            raise BlockError(f"unknown function '{self.text('Operator')}'")

    def out(self, t, u):
        with np.errstate(all="ignore"):
            return [np.asarray(self.fn(*u), float)]


_TRIG = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "sincos",
]


@block(
    "TrigonometricFunction",
    "Math",
    "sin, cos, tan and their inverse and hyperbolic versions; atan2 has two inputs.",
    aliases=("Trigonometry",),
    icon="trig",
)
class TrigonometricFunction(Node):
    params = [P("Operator", "sin", "Function", options=_TRIG)]
    nin = ["if", "Operator", "atan2", 2, 1]
    nout = ["if", "Operator", "sincos", 2, 1]

    def init(self):
        self.op = self.text("Operator")
        if self.op not in _TRIG:
            raise BlockError(f"unknown function '{self.op}'")

    def out(self, t, u):
        with np.errstate(all="ignore"):
            if self.op == "sincos":
                return [np.sin(u[0]), np.cos(u[0])]
            if self.op == "atan2":
                return [np.arctan2(u[0], u[1])]
            name = {
                "asin": "arcsin",
                "acos": "arccos",
                "atan": "arctan",
                "asinh": "arcsinh",
                "acosh": "arccosh",
                "atanh": "arctanh",
            }.get(self.op, self.op)
            return [getattr(np, name)(u[0])]


def _sqrt(u, b):
    op = b.choice("Operator")
    with np.errstate(all="ignore"):
        if op == "signedsqrt":
            return np.sign(u) * np.sqrt(np.abs(u))
        return 1.0 / np.sqrt(u) if op == "rsqrt" else np.sqrt(u)


elementwise(
    "Sqrt",
    "Math",
    "Square root (also signed square root and reciprocal square root).",
    _sqrt,
    [P("Operator", "sqrt", "Function", options=["sqrt", "signedSqrt", "rSqrt"])],
    icon="√u",
)
elementwise(
    "Rounding",
    "Math",
    "Rounds to whole numbers: floor, ceil, round or fix.",
    lambda u, b: {
        "floor": np.floor,
        "ceil": np.ceil,
        "round": lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5),
        "fix": np.trunc,
    }[b.choice("Operator")](u),
    [P("Operator", "floor", "Function", options=["floor", "ceil", "round", "fix"])],
    title="Rounding Function",
    icon="⌊u⌉",
)


@block("MinMax", "Math", "Smallest or largest of its inputs (or of the elements of a single input).", icon="min")
class MinMax(Node):
    params = [P("Function", "min", "Function", options=["min", "max"]), P("Inputs", "1", "Number of inputs", "text")]
    nin = ["int", "Inputs"]

    def out(self, t, u):
        f = np.minimum if self.choice("Function") == "min" else np.maximum
        if len(u) == 1:
            return [np.array([f.reduce(np.ravel(u[0]))])]
        total = u[0]
        for v in u[1:]:
            total = f(total, v)
        return [total]


@block("DotProduct", "Math", "Dot product of two vectors.", icon="u·v")
class DotProduct(Node):
    params = []
    nin = 2

    def out(self, t, u):
        return [np.array([np.sum(u[0] * u[1])])]


@block(
    "Reshape",
    "Math",
    "Changes the shape of a signal: 1-D vector, column, row, or the given [rows columns].",
    icon="[ ]",
)
class Reshape(Node):
    params = [
        P(
            "OutputDimensionality",
            "1-D array",
            "Output shape",
            options=["1-D array", "Column vector (2-D)", "Row vector (2-D)", "Customize"],
        ),
        P("OutputDimensions", "[1,1]", "Dimensions when customized"),
    ]

    def out(self, t, u):
        if self.choice("OutputDimensionality") != "customize":
            return [np.ravel(u[0], order="F")]
        dims = [int(d) for d in self.num("OutputDimensions")]
        if int(np.prod(dims)) != u[0].size:
            raise BlockError(f"an input with {u[0].size} elements cannot be reshaped to {dims}")
        return [sig(np.reshape(np.ravel(u[0], order="F"), dims, order="F"))]


@block(
    "Concatenate",
    "Math",
    "Joins its inputs into one vector, or side by side / on top of each other as matrices.",
    title="Vector Concatenate",
    aliases=("MatrixConcatenate",),
    icon="[a b]",
)
class Concatenate(Node):
    params = [
        P("NumInputs", "2", "Number of inputs", "text"),
        P("Mode", "Vector", "Mode", options=["Vector", "Multidimensional array"]),
        P("ConcatenateDimension", "1", "Dimension to join along (1 = rows, 2 = columns)"),
    ]
    nin = ["int", "NumInputs"]

    def out(self, t, u):
        if self.choice("Mode") == "vector":
            return [np.concatenate([np.ravel(v, order="F") for v in u])]
        dim = int(self.scalar("ConcatenateDimension"))
        parts = [v.reshape(-1, 1) if v.ndim == 1 else v for v in u]
        try:
            return [sig(np.concatenate(parts, axis=dim - 1))]
        except ValueError:
            raise BlockError(
                "the inputs have sizes that cannot be joined: " + ", ".join(str(v.shape) for v in u)
            ) from None


# =====================================================================================
# Logic
# =====================================================================================

_REL = {"==": np.equal, "~=": np.not_equal, "<": np.less, "<=": np.less_equal, ">=": np.greater_equal, ">": np.greater}


def _relop(node: Node, name: str = "Operator"):
    op = node.text(name).replace("!=", "~=")
    if op not in _REL:
        raise BlockError(f"unknown comparison '{op}'")
    return _REL[op]


@block("RelationalOperator", "Logic", "Compares two inputs (==, ~=, <, <=, >=, >); the output is 1 or 0.", icon="a<b")
class RelationalOperator(Node):
    params = [P("Operator", ">=", "Comparison", options=list(_REL))]
    nin = 2
    has_zc = True

    def init(self):
        self.op = _relop(self)

    def out(self, t, u):
        return [self.latched(lambda: self.op(u[0], u[1]).astype(float))]

    def zc(self, t, u):
        return list(np.ravel(u[0] - u[1]))


@block(
    "CompareToConstant",
    "Logic",
    "Compares the input with a constant; the output is 1 or 0.",
    title="Compare To Constant",
    icon="u<c",
)
class CompareToConstant(Node):
    params = [P("relop", "<=", "Comparison", options=list(_REL)), P("const", "3", "Constant")]
    has_zc = True

    def init(self):
        self.op, self.c = _relop(self, "relop"), self.num("const")

    def out(self, t, u):
        return [self.latched(lambda: self.op(u[0], self.c).astype(float))]

    def zc(self, t, u):
        return list(np.ravel(u[0] - self.c))


@block(
    "CompareToZero", "Logic", "Compares the input with zero; the output is 1 or 0.", title="Compare To Zero", icon="u<0"
)
class CompareToZero(CompareToConstant):
    params = [P("relop", "<=", "Comparison", options=list(_REL))]

    def init(self):
        self.op, self.c = _relop(self, "relop"), np.zeros(1)


_LOGIC = ["AND", "OR", "NAND", "NOR", "XOR", "NXOR", "NOT"]


@block(
    "LogicalOperator",
    "Logic",
    "AND, OR, NAND, NOR, XOR, NXOR or NOT of its inputs (non-zero counts as true).",
    aliases=("Logic",),
    icon="&",
)
class LogicalOperator(Node):
    params = [P("Operator", "AND", "Operator", options=_LOGIC), P("Inputs", "2", "Number of inputs", "text")]
    nin = ["if", "Operator", "NOT", 1, ["int", "Inputs"]]

    def init(self):
        self.op = self.text("Operator").upper()
        if self.op not in _LOGIC:
            raise BlockError(f"unknown logical operator '{self.text('Operator')}'")

    def out(self, t, u):
        b = [v != 0 for v in u]
        if self.op == "NOT":
            return [(~b[0]).astype(float)]
        if len(b) == 1:
            b = [np.array([x]) for x in np.ravel(b[0])]
        f = {
            "AND": np.logical_and,
            "NAND": np.logical_and,
            "OR": np.logical_or,
            "NOR": np.logical_or,
            "XOR": np.logical_xor,
            "NXOR": np.logical_xor,
        }[self.op]
        total = b[0]
        for v in b[1:]:
            total = f(total, v)
        if self.op in ("NAND", "NOR", "NXOR"):
            total = ~total
        return [np.asarray(total).astype(float)]


@block(
    "BitwiseOperator",
    "Logic",
    "Bitwise AND, OR, XOR or NOT of whole-number inputs, optionally with a fixed mask.",
    icon="&b",
)
class BitwiseOperator(Node):
    params = [
        P("logicop", "AND", "Operator", options=["AND", "OR", "XOR", "NOT"]),
        ONOFF("UseBitMask", "on", "Use a bit mask"),
        P("BitMask", "217", "Bit mask"),
        P("NumInputPorts", "1", "Number of inputs", "text"),
    ]
    nin = ["if", "UseBitMask", "on", 1, ["int", "NumInputPorts"]]

    def out(self, t, u):
        op = self.text("logicop").upper()
        vals = [np.asarray(np.round(v), dtype=np.int64) for v in u]
        if op == "NOT":
            return [(~vals[0] & 0xFFFFFFFF).astype(float)]
        if self.is_on("UseBitMask"):
            vals.append(np.asarray(np.round(self.num("BitMask")), dtype=np.int64))
        f = {"AND": np.bitwise_and, "OR": np.bitwise_or, "XOR": np.bitwise_xor}[op]
        total = vals[0]
        for v in vals[1:]:
            total = f(total, v)
        return [np.asarray(total, float)]


@block(
    "Switch",
    "Logic",
    "Passes the first input when the middle (control) input meets the condition, else the third.",
    icon="sw",
)
class Switch(Node):
    params = [
        P(
            "Criteria",
            "u2 >= Threshold",
            "Pass first input when",
            options=["u2 >= Threshold", "u2 > Threshold", "u2 ~= 0"],
        ),
        P("Threshold", "0", "Threshold"),
    ]
    nin = 3
    has_zc = True

    def init(self):
        c = self.text("Criteria").replace(" ", "")
        self.mode = ">=" if ">=" in c else (">" if ">" in c else "~=")
        self.th = self.num("Threshold") if self.mode != "~=" else np.zeros(1)

    def out(self, t, u):
        c = self.latched(
            lambda: u[1] >= self.th if self.mode == ">=" else (u[1] > self.th if self.mode == ">" else u[1] != 0)
        )
        return [np.where(c, u[0], u[2]).astype(float)]

    def zc(self, t, u):
        return list(np.ravel(u[1] - self.th))


@block(
    "MultiportSwitch",
    "Logic",
    "Passes the data input chosen by the first (control) input: 1 for the first, 2 for the next...",
    aliases=("MultiPortSwitch",),
    icon="msw",
)
class MultiportSwitch(Node):
    params = [
        P("Inputs", "3", "Number of data inputs", "text"),
        P(
            "DataPortOrder",
            "One-based contiguous",
            "Numbering",
            options=["One-based contiguous", "Zero-based contiguous"],
        ),
    ]
    nin = ["sum", 1, ["int", "Inputs"]]

    def out(self, t, u):
        k = int(math.floor(float(u[0].flat[0]) + 0.5)) - (0 if self.choice("DataPortOrder").startswith("zero") else 1)
        if not 0 <= k < len(u) - 1:
            raise BlockError(f"the control input is {u[0].flat[0]:g}, but there are {len(u) - 1} data inputs")
        return [u[k + 1]]


@block("ManualSwitch", "Routing", "Passes the first or the second input, as chosen in its dialog.", icon="man")
class ManualSwitch(Node):
    params = [P("sw", "1", "Input to pass (1 = upper, 0 = lower)", options=["1", "0"])]
    nin = 2

    def out(self, t, u):
        return [u[0] if self.text("sw") != "0" else u[1]]


@block("HitCrossing", "Logic", "Outputs 1 at the step where the input crosses a level, otherwise 0.", icon="hit")
class HitCrossing(Node):
    params = [
        P("HitCrossingOffset", "0", "Level"),
        P("HitCrossingDirection", "either", "Direction", options=["either", "rising", "falling"]),
    ]
    has_zc = True

    def init(self):
        self.level, self.dir = self.num("HitCrossingOffset"), self.choice("HitCrossingDirection")
        self.prev = None
        self.hit = np.zeros(1)

    def out(self, t, u):
        if not self.sim.minor:
            d = u[0] - self.level
            p = d if self.prev is None else self.prev
            up, down = (p < 0) & (d >= 0), (p > 0) & (d <= 0)
            hit = up if self.dir == "rising" else (down if self.dir == "falling" else up | down | ((d == 0) & (p != 0)))
            self.hit = hit.astype(float)
        return [self.hit]

    def update(self, t, u):
        self.prev = u[0] - self.level

    def zc(self, t, u):
        return list(np.ravel(u[0] - self.level))


# =====================================================================================
# Static nonlinear blocks
# =====================================================================================


@block(
    "Saturation",
    "Nonlinear",
    "Limits the input to the range between a lower and an upper limit.",
    aliases=("Saturate",),
    icon="sat",
)
class Saturation(Node):
    same_shape = True

    params = [P("UpperLimit", "0.5", "Upper limit"), P("LowerLimit", "-0.5", "Lower limit")]
    has_zc = True

    def init(self):
        self.hi, self.lo = self.num("UpperLimit"), self.num("LowerLimit")
        if np.any(self.lo > self.hi):
            raise BlockError("the lower limit is above the upper limit")

    def out(self, t, u):
        v = u[0]
        side = self.latched(
            lambda: np.where(v >= self.hi, 1, np.where(v <= self.lo, -1, 0))
            + np.zeros(np.broadcast(self.hi, self.lo).shape, dtype=int)
        )
        return [np.where(side > 0, self.hi, np.where(side < 0, self.lo, v)).astype(float)]

    def zc(self, t, u):
        both = u[0] + np.zeros(np.broadcast(self.hi, self.lo).shape)
        # an infinite limit is never reached: a large finite distance keeps the crossing search well defined
        return list(np.clip(np.ravel(both - self.hi), -1e100, 1e100)) + list(
            np.clip(np.ravel(both - self.lo), -1e100, 1e100)
        )


@block(
    "SaturationDynamic",
    "Nonlinear",
    "Limits the middle input to the range given by the upper and lower inputs.",
    title="Saturation Dynamic",
    icon="sat",
)
class SaturationDynamic(Node):
    params = []
    nin = 3

    def out(self, t, u):
        return [np.minimum(np.maximum(u[1], u[2]), u[0])]


@block("DeadZone", "Nonlinear", "Outputs zero inside a band; outside, the input shifted by the band edge.", icon="dz")
class DeadZone(Node):
    same_shape = True

    params = [P("LowerValue", "-0.5", "Start of dead zone"), P("UpperValue", "0.5", "End of dead zone")]
    has_zc = True

    def init(self):
        self.lo, self.hi = self.num("LowerValue"), self.num("UpperValue")

    def out(self, t, u):
        v = u[0]
        side = self.latched(
            lambda: np.where(v > self.hi, 1, np.where(v < self.lo, -1, 0)) + 0 * (self.hi + self.lo).astype(int)
        )
        return [np.where(side > 0, v - self.hi, np.where(side < 0, v - self.lo, 0.0))]

    def zc(self, t, u):
        both = u[0] + 0.0 * (self.hi + self.lo)
        return list(np.ravel(both - self.hi)) + list(np.ravel(both - self.lo))


@block("Relay", "Nonlinear", "Switches on above one level and off below another (hysteresis).", icon="relay")
class Relay(Node):
    params = [
        P("OnSwitchValue", "eps", "Switch-on point"),
        P("OffSwitchValue", "eps", "Switch-off point"),
        P("OnOutputValue", "1", "Output when on"),
        P("OffOutputValue", "0", "Output when off"),
    ]
    has_zc = True

    def init(self):
        self.on_at, self.off_at = self.num("OnSwitchValue"), self.num("OffSwitchValue")
        self.on_val, self.off_val = self.num("OnOutputValue"), self.num("OffOutputValue")
        if np.any(self.off_at > self.on_at):
            raise BlockError("the switch-off point must not be above the switch-on point")
        self.on = None

    def out(self, t, u):
        if self.on is None:
            on = u[0] >= self.on_at
        else:
            on = np.where(self.on, u[0] > self.off_at, u[0] >= self.on_at)
        if not self.sim.minor:
            self.on = on
        elif self.on is not None:
            on = np.broadcast_to(self.on, on.shape)
        return [np.where(on, self.on_val, self.off_val).astype(float)]

    def zc(self, t, u):
        on = self.on if self.on is not None else (u[0] >= self.on_at)
        return list(np.ravel(np.where(on, u[0] - self.off_at, u[0] - self.on_at)))


elementwise(
    "Quantizer",
    "Nonlinear",
    "Rounds the input to a multiple of the quantization interval.",
    lambda u, b: b.num("QuantizationInterval") * np.sign(u) * np.floor(np.abs(u) / b.num("QuantizationInterval") + 0.5),
    [P("QuantizationInterval", "0.5", "Quantization interval")],
    icon="quant",
)


@block(
    "CoulombViscousFriction",
    "Nonlinear",
    "Friction: sign(u) * (gain*|u| + offset).",
    title="Coulomb & Viscous Friction",
    icon="fric",
)
class CoulombViscousFriction(Node):
    params = [
        P("offset", "[1 3 2 0]", "Coulomb friction (offset)"),
        P("gain", "1", "Viscous friction coefficient (gain)"),
    ]
    has_zc = True

    def init(self):
        self.offset, self.gain = self.num("offset"), self.num("gain")

    def out(self, t, u):
        v = u[0]
        s = self.latched(lambda: np.sign(v))
        return [np.asarray(self.gain * v + s * self.offset, float)]

    def zc(self, t, u):
        return list(np.ravel(u[0] + 0.0 * self.offset))


@block("RateLimiter", "Nonlinear", "Limits how fast the output may rise and fall.", icon="rate")
class RateLimiter(Node):
    params = [
        P("RisingSlewLimit", "1", "Largest rising rate"),
        P("FallingSlewLimit", "-1", "Largest falling rate (negative)"),
        P("InitialCondition", "0", "Initial output"),
    ]

    def init(self):
        self.up, self.down = self.num("RisingSlewLimit"), self.num("FallingSlewLimit")
        self.t_prev: float | None = None
        self.y_prev = self.num("InitialCondition")

    def _y(self, t, u):
        if self.t_prev is None or t <= self.t_prev:
            return u if self.t_prev is None else self.y_prev + 0.0 * u
        dt = t - self.t_prev
        rate = (u - self.y_prev) / dt
        return np.where(
            rate > self.up, self.y_prev + dt * self.up, np.where(rate < self.down, self.y_prev + dt * self.down, u)
        )

    def out(self, t, u):
        return [np.asarray(self._y(t, u[0]), float)]

    def update(self, t, u):
        self.y_prev, self.t_prev = np.asarray(self._y(t, u[0]), float), t


@block(
    "Backlash",
    "Nonlinear",
    "Play between two parts: the output only follows once the input has crossed the dead band.",
    icon="lash",
)
class Backlash(Node):
    params = [P("BacklashWidth", "1", "Dead band width"), P("InitialOutput", "0", "Initial output")]

    def init(self):
        self.half, self.y_prev = self.num("BacklashWidth") / 2.0, self.num("InitialOutput")

    def _y(self, u):
        return np.where(
            u > self.y_prev + self.half,
            u - self.half,
            np.where(u < self.y_prev - self.half, u + self.half, self.y_prev),
        )

    def out(self, t, u):
        return [np.asarray(self._y(u[0]), float)]

    def update(self, t, u):
        self.y_prev = np.asarray(self._y(u[0]), float)


def _interp1(x, xs, ys, extrap: str):
    if extrap == "clip":
        return np.interp(x, xs, ys)
    i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, xs.size - 2)
    w = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + w * (ys[i + 1] - ys[i])


@block(
    "LookupTable1D",
    "Nonlinear",
    "Interpolates in a table of breakpoints and values.",
    title="1-D Lookup Table",
    aliases=("Lookup", "Lookup1D", "1DLookupTable", "LookupTable"),
    icon="lut",
)
class LookupTable1D(Node):
    params = [
        P("BreakpointsForDimension1", "[1,2,3]", "Breakpoints"),
        P("Table", "tanh([-5:5])", "Table values"),
        P("ExtrapMethod", "Linear", "Outside the table", options=["Linear", "Clip"]),
    ]

    def init(self):
        self.xs, self.ys = self.num("BreakpointsForDimension1"), self.num("Table")
        if self.xs.size != self.ys.size or self.xs.size < 2 or np.any(np.diff(self.xs) <= 0):
            raise BlockError(
                "breakpoints and table need the same length (2 or more), and the breakpoints must increase"
            )
        self.extrap = self.choice("ExtrapMethod")

    def out(self, t, u):
        return [np.asarray(_interp1(u[0], self.xs, self.ys, self.extrap), float)]


@block(
    "LookupTable2D",
    "Nonlinear",
    "Interpolates in a two-dimensional table: first input picks the row, second the column.",
    title="2-D Lookup Table",
    aliases=("Lookup2D", "2DLookupTable"),
    icon="lut2",
)
class LookupTable2D(Node):
    params = [
        P("BreakpointsForDimension1", "[1,2,3]", "Row breakpoints"),
        P("BreakpointsForDimension2", "[1,2,3]", "Column breakpoints"),
        P("Table", "[4 5 6;16 19 20;10 18 23]", "Table"),
        P("ExtrapMethod", "Linear", "Outside the table", options=["Linear", "Clip"]),
    ]
    nin = 2

    def init(self):
        self.r, self.c, self.tab = (
            self.num("BreakpointsForDimension1"),
            self.num("BreakpointsForDimension2"),
            self.mat("Table"),
        )
        if self.tab.shape != (self.r.size, self.c.size) or min(self.tab.shape) < 2:
            raise BlockError(
                f"the table is {self.tab.shape[0]}x{self.tab.shape[1]} but the breakpoints call for "
                f"{self.r.size}x{self.c.size}"
            )
        self.clip = self.choice("ExtrapMethod") == "clip"

    def out(self, t, u):
        a, b = np.broadcast_arrays(u[0], u[1])
        if self.clip:
            a, b = np.clip(a, self.r[0], self.r[-1]), np.clip(b, self.c[0], self.c[-1])
        i = np.clip(np.searchsorted(self.r, a, side="right") - 1, 0, self.r.size - 2)
        j = np.clip(np.searchsorted(self.c, b, side="right") - 1, 0, self.c.size - 2)
        wa = (a - self.r[i]) / (self.r[i + 1] - self.r[i])
        wb = (b - self.c[j]) / (self.c[j + 1] - self.c[j])
        tab = self.tab
        top = tab[i, j] + wb * (tab[i, j + 1] - tab[i, j])
        bot = tab[i + 1, j] + wb * (tab[i + 1, j + 1] - tab[i + 1, j])
        return [np.asarray(top + wa * (bot - top), float)]
