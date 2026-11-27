"""Subsystems that run as one unit: iterator and function-call subsystems, plus their control blocks.

Most subsystems are flattened away before a model runs (see :mod:`.engine`).
Two kinds cannot be: a For or While Iterator subsystem runs its blocks
several times in one time step, and a function-call subsystem runs only when
another block calls it. Such a subsystem becomes one :class:`AtomicSystem`
node. It owns a compiled copy of its inside (a small simulation of its own)
and steps it when it has to.

This module also has the blocks that only make sense with them (For
Iterator, While Iterator, Function-Call Generator, Switch Case, Reset port)
and the helpers that save and restore the memory of blocks, which resettable
subsystems and the "reset states when enabling" option use too.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .library import BlockError, Node, ONOFF, P, block, sig

# -------------------------------------------------------------------------------------
# Saving and restoring what a block remembers
# -------------------------------------------------------------------------------------

# Attributes that describe the block's place in the model, not its memory.
_KEEP = {"sim", "ctx", "inputs", "conds", "rate", "path", "raw", "block", "loop", "in_names", "stamp", "members"}
# What sinks have recorded: a reset must not throw it away.
_LOGS = {"log_t", "log_u", "count", "last_count", "last"}


def _copy(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, deque):
        return deque(np.array(e, copy=True) if isinstance(e, np.ndarray) else e for e in v)
    if isinstance(v, list):
        return [e.copy() if isinstance(e, np.ndarray) else e for e in v]
    if isinstance(v, dict):
        return dict(v)
    return v


def snapshot(node: Node) -> dict:
    """What a block remembers now, to be put back later with :func:`restore`."""
    return {k: _copy(v) for k, v in node.__dict__.items() if k not in _KEEP and k not in _LOGS}


def restore(node: Node, snap: dict) -> None:
    for k, v in snap.items():
        node.__dict__[k] = _copy(v)


# -------------------------------------------------------------------------------------
# Control blocks
# -------------------------------------------------------------------------------------


@block(
    "ForIterator",
    "Routing",
    "Put inside a subsystem: the subsystem runs its blocks N times in every time step; outputs the iteration number.",
    title="For Iterator",
    icon="for",
)
class ForIterator(Node):
    params = [
        P("IterationSource", "internal", "Number of iterations from", options=["internal", "external"]),
        P("IterationLimit", "5", "Number of iterations"),
        P("ResetStates", "held", "States at the start of each time step", options=["held", "reset"]),
        P("IndexMode", "One-based", "Iteration numbering", options=["One-based", "Zero-based"]),
        ONOFF("ShowIterationVariable", "on", "Output the iteration number"),
    ]
    nin = ["if", "IterationSource", "external", 1, 0]
    nout = ["if", "ShowIterationVariable", "on", 1, 0]
    df = False
    sample = "cont"
    time_varying = True

    def out(self, t, u):
        k = getattr(self.sim, "iteration", 1)
        if self.choice("IndexMode") == "zerobased":
            k -= 1
        return [np.array([float(k)])] if self.n_out() else []


@block(
    "WhileIterator",
    "Routing",
    "Put inside a subsystem: the subsystem repeats its blocks in a time step while the condition input is true.",
    title="While Iterator",
    icon="while",
)
class WhileIterator(Node):
    params = [
        P("MaxIters", "5", "Most iterations per time step (-1: no limit)"),
        P("WhileBlockType", "while", "Kind of loop", options=["while", "do-while"]),
        P("ResetStates", "held", "States at the start of each time step", options=["held", "reset"]),
        ONOFF("ShowIterationPort", "off", "Output the iteration number"),
    ]
    nin = ["if", "WhileBlockType", "do-while", 1, 2]
    nout = ["if", "ShowIterationPort", "on", 1, 0]
    df = False
    sample = "cont"
    is_sink = True
    time_varying = True

    def out(self, t, u):
        return [np.array([float(getattr(self.sim, "iteration", 1))])] if self.n_out() else []


@block(
    "FunctionCallGenerator",
    "Sources",
    "Calls the function-call subsystem wired to it at every sample time (a number of times in a row).",
    title="Function-Call Generator",
    aliases=("FunctionCallGenerator", "Function-CallGenerator"),
    icon="f()",
)
class FunctionCallGenerator(Node):
    params = [P("SampleTime", "1", "Sample time (s)"), P("NumberOfIterations", "1", "Calls at each sample time")]
    nin = 0
    sample = "param"

    def init(self):
        self.n = int(self.scalar("NumberOfIterations"))
        if self.n < 0:
            raise BlockError("the number of calls must not be negative")

    def out(self, t, u):
        return [np.array([float(self.n)])]


@block(
    "SwitchCase",
    "Logic",
    "Compares its input with lists of values (cases) and switches on the action subsystem of the matching case.",
    title="Switch Case",
    icon="case",
)
class SwitchCase(Node):
    params = [
        P("CaseConditions", "{1,2}", "Cases: a cell array of values or value lists, e.g. {1,[2 3]}", "text"),
        ONOFF("ShowDefaultCase", "on", "Default output for any other value"),
    ]
    nin = 1
    nout = ["sum", ["cell", "CaseConditions"], ["if", "ShowDefaultCase", "on", 1, 0]]
    time_varying = True

    def init(self):
        from ..values import Cell

        v = self.value("CaseConditions")
        items = v.data.ravel(order="F").tolist() if isinstance(v, Cell) else [v]
        self.cases = []
        for item in items:
            a = sig(item)
            if a.size == 0 or np.any(a != np.round(a)):
                raise BlockError("each case must be one or more whole numbers")
            self.cases.append(set(int(x) for x in a.ravel()))
        self.default = self.is_on("ShowDefaultCase")
        self.choice_now = None

    def out(self, t, u):
        if self.sim.minor and self.choice_now is not None:
            pick = self.choice_now
        else:
            x = np.ravel(u[0])
            if x.size != 1:
                raise BlockError("the input must be a single number")
            v = int(np.trunc(x[0]))  # values in between are cut toward zero, like a conversion to an integer
            pick = next((k for k, c in enumerate(self.cases) if v in c), len(self.cases))
            if not self.sim.minor:
                self.choice_now = pick
        n = len(self.cases) + (1 if self.default else 0)
        return [np.array([1.0 if k == pick else 0.0]) for k in range(n)]


@block(
    "ResetPort",
    "Routing",
    "Put inside a subsystem: the blocks inside start again from their initial states when the reset input fires.",
    title="Reset",
    icon="rst",
)
class ResetPort(Node):
    params = [
        P("ResetTriggerType", "rising", "Reset on", options=["rising", "falling", "either", "level", "level hold"]),
    ]
    nin = 0
    nout = 0
    virtual = True


# -------------------------------------------------------------------------------------
# The node that runs a subsystem as a unit
# -------------------------------------------------------------------------------------


def atomic_kind(inner) -> tuple[str, Any] | None:
    """('for' | 'while' | 'call', control block) when a subsystem must run as one unit, else None."""
    from .library import find_type

    for b in inner.blocks:
        t = getattr(find_type(b.type), "type", "")
        if t == "ForIterator":
            return "for", b
        if t == "WhileIterator":
            return "while", b
        if t == "TriggerPort" and _is_call(b):
            return "call", b
    return None


def _is_call(b) -> bool:
    return b.params.get("TriggerType", "").replace("-", "").replace(" ", "").lower() == "functioncall"


class AtomicSystem(Node):
    """A For/While Iterator or function-call subsystem, run as one block with its own compiled inside."""

    def __init__(self, path, blk, inner, ctx, comp, kind, control):
        super().__init__(path, {}, ctx)
        from .engine import Context, compile_model
        from .model import Model

        self.kind, self.control, self.blk = kind, control, blk
        self.outer = comp
        sub = Model(blk.name)
        sub.root = inner
        sub.config = dict(comp.model.config)
        shared = {"t0": ctx.shared["t0"], "tf": ctx.shared["tf"], "ext": {}, "cache": {}, "n": 0}
        ictx = Context(ctx.interp, ctx.frame, dict(ctx.scope), shared)
        self.icomp = compile_model(sub, ctx.interp, ctx.frame, strict=False, ctx=ictx, prefix=path)
        for d in self.icomp.diagnostics:
            comp.diagnostics.append(d)
        from .blocks_basic import Inport, Outport

        nodes = self.icomp.nodes
        self.ins = {n.port: n for n in nodes if isinstance(n, Inport)}
        self.outs = sorted((n for n in nodes if isinstance(n, Outport)), key=lambda n: int(float(n.text("Port") or 1)))
        self.n_data = _count(inner, "Inport")
        self.n_outs = _count(inner, "Outport")
        self.ctl = next((n for n in nodes if isinstance(n, (ForIterator, WhileIterator))), None)
        self.u_now: list = [np.zeros(1)] * self.n_data
        self.inner = None
        self.iteration = 1
        self.reset_each = control.params.get("ResetStates", "held").strip().lower() == "reset"
        for n in nodes:
            if n.nx:
                comp.problem(
                    "error",
                    f"'{n.path}': blocks with continuous states cannot be inside "
                    f"{'an iterator' if kind != 'call' else 'a function-call'} subsystem.",
                    n.path,
                    kind="param",
                )

    # structure
    def n_in(self):
        return self.n_data + (1 if self.kind == "call" else 0)

    def n_out(self):
        return self.n_outs

    def is_df(self, port):
        return True

    def sample_time(self):
        return None

    def init(self):
        def port(k):
            return lambda t: self.u_now[k - 1] if k - 1 < len(self.u_now) else np.zeros(1)

        for k in self.ins:
            self.icomp.ctx.shared["ext"][k] = port(k)

    def _start(self, shapes) -> None:
        if self.icomp.errors():
            first = self.icomp.errors()[0]
            raise BlockError(first["message"])
        self.u_now = [np.zeros(s) if s is not None else np.zeros(1) for s in (shapes or [None] * self.n_data)]
        self.u_now = self.u_now[: self.n_data] + [np.zeros(1)] * (self.n_data - len(self.u_now))
        inner = _InnerSim(self.icomp, self)
        self.inner = inner
        self.first = snapshot_all(inner.order)
        self.held = [self._initial(k) for k in range(self.n_outs)]
        self.y = list(self.held)

    def _initial(self, k: int) -> np.ndarray:
        n = self.outs[k] if k < len(self.outs) else None
        v = np.zeros(1)
        if n is not None:
            got = self.inner.gather(n)[0]
            v = np.zeros(np.shape(got)) if np.size(got) else np.zeros(1)
            init = n.text("InitialOutput")
            if init not in ("", "[]"):
                try:
                    v = np.array(np.broadcast_to(sig(n.ctx.evaluate(init)), v.shape), dtype=float)
                except Exception:
                    v = sig(n.ctx.evaluate(init))
        return v

    def setup(self, shapes):
        data = list(shapes[: self.n_data]) if shapes else None
        if self.inner is None or (data is not None and data != getattr(self, "_shapes", None)):
            self._shapes = data
            before = [np.shape(v) for v in getattr(self, "held", [])]
            self._start(data)
            return [np.shape(v) for v in self.held] != before
        return False

    def reset_states(self) -> None:
        if self.inner is not None:
            restore_all(self.inner.order, self.first)
            self.held = [self._initial(k) for k in range(self.n_outs)]

    # running
    def _run_once(self, t: float, logging: bool) -> None:
        sim = self.inner
        sim.step_no += 1
        sim.hits = set(sim.rates)
        sim.eval(t, sim.x0, False, logging)
        step = sim.step_no
        for n in sim.order:
            if n.stamp == step:
                n.update(t, sim.gather(n))
        sim.logging = False

    def _peek(self, t: float, node: Node, port: int) -> np.ndarray:
        """The value at an input of a block inside, without changing anything."""
        sim = self.inner
        saved = snapshot_all(sim.order)
        try:
            sim.eval(t, sim.x0, True)
            return sim.gather(node)[port]
        finally:
            restore_all(sim.order, saved)

    def _limit(self, t: float) -> int:
        c = self.ctl
        if isinstance(c, ForIterator):
            if c.choice("IterationSource") == "external":
                v = np.ravel(self._peek(t, c, 0))
                return max(int(round(float(v[0]))) if v.size else 0, 0)
            return max(int(c.scalar("IterationLimit")), 0)
        return 0

    def _loop(self, t: float, logging: bool) -> None:
        sim = self.inner
        if self.reset_each:
            restore_all(sim.order, self.first)
        if self.kind == "for":
            n = self._limit(t)
            for k in range(1, n + 1):
                self.iteration = sim.iteration = k
                self._run_once(t, logging and k == n)
            ran = n > 0
        else:
            c = self.ctl
            most = int(c.scalar("MaxIters"))
            do_while = c.choice("WhileBlockType") == "dowhile"
            go = True if do_while else bool(np.any(self._peek(t, c, 1) != 0))
            k = 0
            while go:
                k += 1
                if most >= 0 and k > most:
                    break
                self.iteration = sim.iteration = k
                self._run_once(t, logging)
                go = bool(np.any(sim.gather(c)[0] != 0))
                if most < 0 and k >= 10_000_000:
                    raise BlockError("the While Iterator has run ten million times in one step without stopping")
            ran = k > 0
        if ran:
            self.held = [np.array(sim.gather(o)[0], dtype=float, copy=True) for o in self.outs]

    def out(self, t, u):
        if self.inner is None:
            self._start([np.shape(v) for v in u[: self.n_data]])
        self.u_now = [np.asarray(v, float) for v in u[: self.n_data]]
        outer = self.sim
        if self.kind == "call":
            link = self.inputs[-1] if self.inputs else None
            calls = 0
            if link is not None and not outer.minor and link[0].stamp == outer.step_no and not outer.initializing:
                v = np.ravel(u[-1])
                calls = int(round(float(v[0]))) if v.size else 0
            for _ in range(max(calls, 0)):
                self._run_once(t, outer.logging)
            if calls > 0:
                self.held = [np.array(self.inner.gather(o)[0], dtype=float, copy=True) for o in self.outs]
            return list(self.held)
        if outer.minor or outer.initializing:
            saved = snapshot_all(self.inner.order)
            held = self.held
            try:
                self._loop(t, False)
                return list(self.held)
            finally:
                restore_all(self.inner.order, saved)
                self.held = held
        self._loop(t, outer.logging)
        return list(self.held)

    def all_nodes(self) -> list:
        out = []
        for n in self.inner.order if self.inner is not None else []:
            out.append(n)
            if isinstance(n, AtomicSystem):
                out += n.all_nodes()
        return out


def _count(inner, type_name: str) -> int:
    from .library import find_type

    return sum(1 for b in inner.blocks if getattr(find_type(b.type), "type", "") == type_name)


def snapshot_all(nodes: list) -> list:
    out = []
    for n in nodes:
        out.append(snapshot(n))
        if isinstance(n, AtomicSystem) and n.inner is not None:
            out.append(snapshot_all(n.inner.order))
    return out


def restore_all(nodes: list, snaps: list) -> None:
    it = iter(snaps)
    for n in nodes:
        restore(n, next(it))
        if isinstance(n, AtomicSystem) and n.inner is not None:
            restore_all(n.inner.order, next(it))


def _inner_sim_class():
    from .engine import Sim

    class InnerSim(Sim):
        """The simulation of the inside of an atomic subsystem. Stops and notes go to the model around it."""

        def __init__(self, comp, owner):
            self.owner = owner
            self.iteration = 1
            super().__init__(comp)

        def request_stop(self, path):
            self.owner.sim.request_stop(path)

        def note(self, message):
            self.owner.outer.problem("warning", message, None, kind="run")

    return InnerSim


_INNER = None


def _InnerSim(comp, owner):
    global _INNER
    if _INNER is None:
        _INNER = _inner_sim_class()
    return _INNER(comp, owner)


@block(
    "ModelReference",
    "Routing",
    "Runs another model as a block: the In and Out blocks of its top level become the ports.",
    title="Model",
    aliases=("Model",),
    icon="mdl",
)
class ModelReference(Node):
    params = [P("ModelName", "", "Referenced model (name of an open model or a model file)", "text")]
    nin = "system"
    nout = "system"
    virtual = True
