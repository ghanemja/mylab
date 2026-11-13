"""Block types: the base class, the registry and the catalog used by the editor.

Every block type is a subclass of :class:`Node` registered with
:func:`block`. An instance is one block in a running simulation. Signals are
NumPy float arrays: 1-D for scalars and vectors, 2-D for matrices.

The simulator calls, in this order:

``init()``      once, to read parameters (``self.num('Gain')`` evaluates the
                MATLAB expression typed into the parameter).
``setup(shapes)`` to size continuous states once the input shapes are known.
``out(t, u)``   block outputs (a list of arrays) from the inputs ``u``. Called
                in *major* steps (results are logged, modes may be latched)
                and in *minor* steps (trial evaluations made by the solver;
                ``self.sim.minor`` is true and nothing may be remembered).
``der(t, u)``   time derivative of the continuous states ``self.x``.
``update(t, u)`` end of a major step: advance discrete states.
``zc(t, u)``    values whose sign changes mark an event the solver should
                locate.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

import numpy as np

INF = math.inf
TYPES: dict[str, type["Node"]] = {}
ALIASES: dict[str, str] = {}
CATEGORIES = ["Sources", "Sinks", "Continuous", "Discrete", "Math", "Logic", "Nonlinear", "Routing", "User-defined"]


class BlockError(Exception):
    """A problem in one block (bad parameter, wrong input size...)."""


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class P:
    """One block parameter: name, default text, kind and a label for the dialog."""

    __slots__ = ("name", "default", "kind", "label", "options")

    def __init__(self, name: str, default: str = "0", label: str = "", kind: str = "expr", options: list | None = None):
        self.name = name
        self.default = default
        self.kind = "enum" if options else kind
        self.label = label or name
        self.options = options or []

    def to_json(self) -> dict:
        d = {"name": self.name, "default": self.default, "kind": self.kind, "label": self.label}
        if self.options:
            d["options"] = self.options
        return d


def ONOFF(name: str, default: str, label: str) -> P:
    return P(name, default, label, "bool")


def SAMPLE(default: str = "-1") -> P:
    return P("SampleTime", default, "Sample time (s); -1 takes it from the input, 0 is continuous")


def block(type_name: str, category: str, doc: str, title: str = "", aliases: tuple = (), icon: str = ""):
    def deco(cls):
        cls.type = type_name
        cls.category = category
        cls.doc = doc
        cls.title = title or re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", type_name)
        cls.icon = icon
        TYPES[type_name] = cls
        for a in (type_name, cls.title) + tuple(aliases):
            ALIASES.setdefault(_key(a), type_name)
        return cls
    return deco


def find_type(name: str) -> type["Node"] | None:
    """Look a block type up by any familiar spelling: 'Transfer Fcn', 'library/Continuous/TransferFcn'..."""
    last = name.replace("\n", " ").split("/")[-1]
    t = ALIASES.get(_key(last))
    return TYPES.get(t) if t else None


def count_ports(rule: Any, params: dict) -> int:
    """Number of ports from a port rule (the editor implements the same rules in JavaScript)."""
    if isinstance(rule, int):
        return rule
    if isinstance(rule, str):  # "system": counted from the Inport/Outport blocks inside
        return 0
    kind = rule[0]
    if kind == "sum":
        return sum(count_ports(r, params) for r in rule[1:])
    if kind == "if":
        want = rule[2] if isinstance(rule[2], list) else [rule[2]]
        hit = _key(str(params.get(rule[1], ""))) in [_key(w) for w in want]
        return count_ports(rule[3] if hit else rule[4], params)
    val = str(params.get(rule[1], "")).strip()
    if kind == "int":
        try:
            return max(int(float(val)), 0)
        except ValueError:
            return 1
    if kind == "signs":  # '+-+' -> 3, '3' -> 3, '[2 1]' -> 2
        if re.fullmatch(r"\d+", val):
            return int(val)
        if val.startswith("["):
            return max(len(val.strip("[]").replace(",", " ").split()), 1)
        return max(len(val.replace("|", "").replace(" ", "")), 1)
    if kind == "csv":
        return max(len([s for s in val.split(",") if s.strip()]), 1)
    if kind == "sig":  # ports of a function block from its first 'function' line
        ins, outs = signature(val)
        return len(ins) if rule[2] == "in" else len(outs)
    raise ValueError(f"bad port rule {rule!r}")


def signature(code: str) -> tuple[list[str], list[str]]:
    """Input and output names of the first function line in ``code``."""
    m = re.search(r"^\s*function\s+(?:\[([^\]]*)\]\s*=|(\w+)\s*=)?\s*\w+\s*(?:\(([^)]*)\))?", code, re.M)
    if not m:
        return ["u"], ["y"]
    outs = [s for s in re.split(r"[,\s]+", (m.group(1) or m.group(2) or "").strip()) if s]
    ins = [s for s in re.split(r"[,\s]+", (m.group(3) or "").strip()) if s]
    return ins, outs


def sig(v: Any) -> np.ndarray:
    """A MATLAB value or Python number as a signal: 1-D unless it is a true matrix."""
    a = np.asarray(v)
    if a.dtype.kind in "US" or a.dtype == object:
        raise BlockError("a number was expected, not text")
    if np.iscomplexobj(a):
        if np.any(a.imag != 0):
            raise BlockError("complex values are not supported in block diagrams")
        a = a.real
    a = a.astype(float)
    if a.ndim == 2 and min(a.shape) > 1:
        return a
    return a.reshape(-1) if a.size else np.zeros(0)


def to_value(a: np.ndarray) -> np.ndarray:
    """A signal as a MATLAB value: scalars 1x1, vectors as columns."""
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim < 2 else a


class Node:
    type = ""
    title = ""
    category = ""
    doc = ""
    icon = ""
    params: list[P] = []
    nin: Any = 1
    nout: Any = 1
    df: Any = True           # direct feedthrough: True, False, or a list per input
    sample = "inherit"       # "inherit", "cont", "const" or "param" (reads SampleTime)
    virtual = False
    nx = 0
    has_zc = False
    is_sink = False
    time_varying = False     # the output can change even while all inputs are constant

    def __init__(self, path: str, raw: dict, ctx: Any):
        self.path = path
        self.raw = {p.name: p.default for p in self.params}
        self.raw.update(raw)
        self.ctx = ctx
        self.sim: Any = None
        self.x = np.zeros(0)
        self.y: list[np.ndarray] = []
        self.inputs: list = []
        self.conds: list = []
        self.rate: Any = None
        self.stamp = -1
        self._cache: dict[str, Any] = {}
        self._mode: Any = None

    # -- parameters -------------------------------------------------------------------
    def text(self, name: str) -> str:
        return str(self.raw.get(name, "")).strip()

    def is_on(self, name: str) -> bool:
        return self.text(name).lower() in ("on", "true", "1", "yes")

    def value(self, name: str) -> Any:
        """The parameter's MATLAB value (the expression is evaluated in the workspace)."""
        if name not in self._cache:
            expr = self.text(name)
            if expr == "":
                raise BlockError(f"parameter '{name}' is empty")
            try:
                self._cache[name] = self.ctx.evaluate(expr)
            except Exception as exc:
                msg = getattr(exc, "message", None) or str(exc)
                raise BlockError(f"parameter {name} = '{expr}' cannot be evaluated: {msg}") from None
        return self._cache[name]

    def num(self, name: str) -> np.ndarray:
        try:
            return sig(self.value(name))
        except BlockError as exc:
            if "cannot be evaluated" in str(exc):
                raise
            raise BlockError(f"parameter {name} = '{self.text(name)}': {exc}") from None

    def mat(self, name: str) -> np.ndarray:
        """Parameter as a 2-D matrix (keeps row/column orientation)."""
        self.num(name)
        a = np.asarray(self.value(name))
        a = a.real.astype(float) if np.iscomplexobj(a) else a.astype(float)
        return a.reshape(1, -1) if a.ndim < 2 else a

    def scalar(self, name: str) -> float:
        a = self.num(name)
        if a.size != 1:
            raise BlockError(f"parameter {name} = '{self.text(name)}' must be a single number")
        return float(a.flat[0])

    def choice(self, name: str) -> str:
        """An option parameter, lower case without spaces or punctuation."""
        return _key(self.text(name))

    # -- structure --------------------------------------------------------------------
    def n_in(self) -> int:
        return count_ports(self.nin, self.raw)

    def n_out(self) -> int:
        return count_ports(self.nout, self.raw)

    def is_df(self, port: int) -> bool:
        return self.df[port] if isinstance(self.df, (list, tuple)) else bool(self.df)

    def sample_time(self) -> Any:
        """None = continuous, 'inherit', 'const' or (period, offset)."""
        if self.sample == "cont":
            return None
        if self.sample in ("const", "inherit"):
            return self.sample
        return self.read_sample("SampleTime")

    def read_sample(self, name: str, default: Any = "inherit") -> Any:
        ts = self.num(name) if self.text(name) else np.array([-1.0])
        if ts.size not in (1, 2):
            raise BlockError(f"parameter {name} must be a period or [period offset]")
        period, offset = float(ts[0]), float(ts[1]) if ts.size == 2 else 0.0
        if period == -1:
            return default
        if period == 0:
            return None
        if math.isinf(period):
            return "const"
        if period < 0 or offset < 0 or offset >= period:
            raise BlockError(f"parameter {name} = '{self.text(name)}' is not a valid sample time")
        return (period, offset)

    @property
    def period(self) -> float:
        """Sample period of a discrete block."""
        if not isinstance(self.rate, tuple):
            raise BlockError("this block needs a discrete sample time (a positive SampleTime, or a discrete input)")
        return self.rate[0]

    def latched(self, compute: Callable) -> Any:
        """A mode (which side of a limit the input is on, a switch position...).

        It is worked out afresh in major steps. While the solver locates zero
        crossings, minor steps reuse the mode of the last major step, so the
        equations stay smooth inside a step and change only at the located event.
        """
        sim = self.sim
        if sim.minor and sim.latch and self._mode is not None:
            return self._mode
        mode = compute()
        if not sim.minor:
            self._mode = mode
        return mode

    # -- behaviour (defaults) ---------------------------------------------------------
    def init(self) -> None:
        pass

    def setup(self, shapes: list | None) -> bool:
        return False

    def out(self, t: float, u: list) -> list:
        return []

    def der(self, t: float, u: list) -> np.ndarray:
        return np.zeros(self.nx)

    def update(self, t: float, u: list) -> None:
        pass

    def zc(self, t: float, u: list) -> list:
        return []

    def next_break(self, t: float) -> float:
        return INF


def elementwise(type_name: str, category: str, doc: str, fn: Callable, params: list | None = None, zc: Callable | None = None,
                title: str = "", aliases: tuple = (), icon: str = ""):
    """Register a block with one input and one output: y = fn(u, block)."""
    plist = params or []
    zc_fn = zc

    class Elementwise(Node):
        params = plist
        has_zc = zc_fn is not None

        def out(self, t, u):
            return [np.asarray(fn(u[0], self), dtype=float)]

        def zc(self, t, u):
            return list(np.ravel(zc_fn(u[0], self))) if zc_fn else []

    Elementwise.__name__ = type_name
    return block(type_name, category, doc, title, aliases, icon)(Elementwise)


def catalog() -> dict:
    """Everything the editor needs to know about the block types, as plain data."""
    out = []
    for name, cls in TYPES.items():
        out.append({
            "type": name, "title": cls.title, "category": cls.category, "doc": cls.doc, "icon": cls.icon,
            "params": [p.to_json() for p in cls.params], "in": cls.nin, "out": cls.nout,
            "sink": cls.is_sink,
        })
    out.sort(key=lambda d: (CATEGORIES.index(d["category"]), d["title"]))
    return {"categories": CATEGORIES, "blocks": out}


from . import blocks_basic, blocks_dynamic, blocks_more  # noqa: E402,F401  (register the block types)
