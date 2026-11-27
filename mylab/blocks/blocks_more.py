"""Signal routing, subsystem ports and user-defined function blocks."""

from __future__ import annotations

import re

import numpy as np

from .library import BlockError, Node, ONOFF, P, _key, block, sig, signature, to_value


@block("Mux", "Routing", "Joins its inputs into one vector signal.", icon="mux")
class Mux(Node):
    params = [P("Inputs", "2", "Number of inputs", "text")]
    nin = ["signs", "Inputs"]

    def out(self, t, u):
        return [np.concatenate([np.ravel(v, order="F") for v in u]) if u else np.zeros(0)]


@block("Demux", "Routing", "Splits a vector signal into its elements (or into parts of given widths).", icon="demux")
class Demux(Node):
    params = [P("Outputs", "2", "Number of outputs, or a vector of widths", "text")]
    nout = ["signs", "Outputs"]

    def init(self):
        s = self.text("Outputs")
        self.widths = [int(w) for w in self.num("Outputs")] if s.startswith("[") else None
        self.n = len(self.widths) if self.widths else int(self.scalar("Outputs"))

    def out(self, t, u):
        v = np.ravel(u[0], order="F")
        widths = self.widths
        if widths is None:
            if v.size % self.n and v.size != 1:
                raise BlockError(f"an input with {v.size} elements cannot be split evenly into {self.n} outputs")
            widths = [max(v.size // self.n, 1)] * self.n
            if v.size == 1 and self.n > 1:
                return [v] * self.n
        if sum(widths) != v.size:
            raise BlockError(f"the input has {v.size} elements but the output widths add up to {sum(widths)}")
        res, k = [], 0
        for w in widths:
            res.append(v[k : k + w])
            k += w
        return res


def _split_top(text: str) -> list[str]:
    """Split at commas that are not inside brackets."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


_INDEX_KINDS = {
    "selectall": "all",
    "assignall": "all",
    "indexvectordialog": "vec",
    "indexvectorport": "vecport",
    "startingindexdialog": "start",
    "startingindexport": "startport",
}


class _Indexed(Node):
    """Shared index handling of Selector and Assignment, for any number of dimensions.

    Each dimension has an option: all elements, an index vector (typed in, or from an input
    port), or a starting index (typed in, or from a port) followed by a number of elements.
    """

    def read_indices(self):
        self.ndim = int(self.scalar("NumberOfDimensions"))
        if self.ndim < 1:
            raise BlockError("the number of dimensions must be at least 1")
        opts = [_key(o) for o in _split_top(self.text("IndexOptions"))]
        exprs = _split_top(self.text("Indices"))
        sizes = _split_top(self.text("OutputSizes")) if "OutputSizes" in self.raw else []
        self.kinds, self.idx, self.lengths = [], [], []
        base = 0 if self.zero_based else 1
        for d in range(self.ndim):
            opt = opts[d] if d < len(opts) and opts[d] else (opts[-1] if len(opts) == 1 and d else "indexvectordialog")
            kind = _INDEX_KINDS.get(opt)
            if kind is None:
                raise BlockError(f"the index option '{_split_top(self.text('IndexOptions'))[d]}' is not known")
            self.kinds.append(kind)
            value = None
            if kind in ("vec", "start"):
                expr = exprs[d] if d < len(exprs) and exprs[d] else "1"
                try:
                    value = np.round(np.ravel(sig(self.ctx.evaluate(expr)))).astype(int) - base
                except Exception as exc:
                    msg = getattr(exc, "message", exc)
                    raise BlockError(f"the index '{expr}' cannot be evaluated: {msg}") from None
                if kind == "start" and value.size != 1:
                    raise BlockError(f"a starting index must be one number, not '{expr}'")
            self.idx.append(value)
            length = None
            if kind in ("start", "startport") and d < len(sizes) and sizes[d]:
                try:
                    length = int(round(float(np.ravel(sig(self.ctx.evaluate(sizes[d])))[0])))
                except Exception as exc:
                    raise BlockError(
                        f"the output size '{sizes[d]}' cannot be evaluated: {getattr(exc, 'message', exc)}"
                    ) from None
            self.lengths.append(length)
        self.port_dims = [d for d, k in enumerate(self.kinds) if k.endswith("port")]
        self.port_index = bool(self.port_dims)

    @property
    def zero_based(self):
        return self.choice("IndexMode") == "zerobased"

    def shaped(self, a: np.ndarray) -> np.ndarray:
        """The signal with exactly ndim dimensions (a vector is a column)."""
        a = np.asarray(a, dtype=float)
        if self.ndim == 1:
            return np.ravel(a, order="F")
        if a.ndim <= 1:
            a = a.reshape(-1, 1)
        if a.ndim < self.ndim:
            a = a.reshape(a.shape + (1,) * (self.ndim - a.ndim))
        elif a.ndim > self.ndim:
            raise BlockError(f"the signal has {a.ndim} dimensions but the block is set up for {self.ndim}")
        return a

    def key(self, a: np.ndarray, ports: list, lengths: list | None = None) -> tuple:
        """Index arrays (one per dimension) for the shaped signal ``a``; ``ports`` are the index inputs."""
        base = 0 if self.zero_based else 1
        arrays = []
        it = iter(ports)
        for d, kind in enumerate(self.kinds):
            n = a.shape[d]
            if kind == "all":
                i = np.arange(n)
            elif kind == "vec":
                i = self.idx[d]
            elif kind == "vecport":
                i = np.round(np.ravel(next(it))).astype(int) - base
            else:
                s0 = int(self.idx[d][0]) if kind == "start" else int(round(float(np.ravel(next(it))[0]))) - base
                length = (lengths[d] if lengths else None) or self.lengths[d] or 1
                i = s0 + np.arange(length)
            if i.size and (i.min() < 0 or i.max() >= n):
                what = f"{a.size} elements" if self.ndim == 1 else "x".join(str(k) for k in a.shape)
                raise BlockError(f"an index is outside the input, which has {what}")
            arrays.append(i)
        return np.ix_(*arrays) if self.ndim > 1 else (arrays[0],)

    def want_min(self, port):
        i = self.idx[0] if self.ndim == 1 and self.kinds[0] == "vec" else None
        return (int(i.max()) + 1,) if isinstance(i, np.ndarray) and i.size and i.max() > 0 else None


_INDEX_PARAMS = [
    P("NumberOfDimensions", "1", "Number of dimensions"),
    P("IndexMode", "One-based", "Index numbering", options=["One-based", "Zero-based"]),
    P(
        "IndexOptions",
        "Index vector (dialog)",
        "Index option per dimension: Select all, Index vector (dialog), Index vector (port), Starting index "
        "(dialog), Starting index (port)",
        "text",
    ),
    P("Indices", "[1 3]", "Indices (or starting indices), one expression per dimension", "text"),
]


@block("Selector", "Routing", "Picks elements of a signal by index: a vector, rows and columns, or more dimensions.", icon="u(i)")  # noqa: E501
class Selector(_Indexed):
    params = _INDEX_PARAMS + [
        P("OutputSizes", "1", "Number of elements per dimension after a starting index", "text"),
        P("InputPortWidth", "-1", "Width of the input, if known (-1: any)"),
    ]
    nin = ["sum", 1, ["ports", "IndexOptions"]]

    def want_in(self, port):
        if port or self.ndim != 1:
            return None
        try:
            n = int(self.scalar("InputPortWidth"))
        except BlockError:
            return None
        return (n,) if n > 1 else None

    def init(self):
        self.read_indices()

    def out(self, t, u):
        a = self.shaped(u[0])
        k = self.key(a, u[1:])
        y = a[k]
        if self.ndim == 1:
            return [np.ravel(y)]
        return [sig(y)]


@block("Assignment", "Routing", "Replaces chosen elements of a signal (Y0) with the values of another input (U).", icon="y(i)=u")  # noqa: E501
class Assignment(_Indexed):
    params = _INDEX_PARAMS + [
        P(
            "OutputInitialize",
            "Initialize using input port <Y0>",
            "Start from",
            options=["Initialize using input port <Y0>", "Specify size for each dimension"],
        ),
        P("OutputSizes", "1", "Size of the output per dimension (when it is not taken from Y0)", "text"),
    ]
    nin = ["sum", ["if", "OutputInitialize", "Specify size for each dimension", 1, 2], ["ports", "IndexOptions"]]

    def init(self):
        self.read_indices()
        self.has_y0 = not self.choice("OutputInitialize").startswith("specify")
        self.prev = None
        if not self.has_y0:
            texts = _split_top(self.text("OutputSizes"))
            sizes = [int(round(float(np.ravel(sig(self.ctx.evaluate(e)))[0]))) for e in texts]
            sizes = sizes + [1] * (self.ndim - len(sizes))
            self.size0 = tuple(sizes[: self.ndim]) if self.ndim > 1 else (int(np.prod(sizes)),)

    def out(self, t, u):
        if self.has_y0:
            y0, val, ports = u[0], u[1], u[2:]
            # inside a For or While Iterator subsystem, later iterations go on from the previous result
            if getattr(self.sim, "iteration", 1) > 1 and self.prev is not None:
                y0 = self.prev
        else:
            val, ports = u[0], u[1:]
            y0 = self.prev if getattr(self.sim, "iteration", 1) > 1 and self.prev is not None else np.zeros(self.size0)
        y = self.shaped(y0).copy()
        vs = np.asarray(val, dtype=float)
        lengths = None
        if any(k in ("start", "startport") for k in self.kinds):
            vshape = self.shaped(vs).shape if vs.size > 1 else (1,) * self.ndim
            lengths = [vshape[d] if d < len(vshape) else 1 for d in range(self.ndim)]
        k = self.key(y, ports, lengths)
        target = y[k]
        if vs.size == 1:
            y[k] = float(vs.ravel()[0])
        elif vs.size == target.size:
            y[k] = np.reshape(np.ravel(vs, order="F"), target.shape, order="F")
        else:
            raise BlockError(f"the value has {vs.size} elements but {target.size} are to be replaced")
        if not self.sim.minor:
            self.prev = y.copy()
        if self.ndim == 1:
            return [y]
        return [sig(y)]


@block(
    "BusCreator",
    "Routing",
    "Groups signals into a bus under the names of their lines (or of a bus object); a Bus Selector picks them out.",
    icon="bus",
)
class BusCreator(Node):
    params = [
        P("Inputs", "2", "Number of inputs", "text"),
        P("OutDataTypeStr", "Inherit: auto", "Bus object for the element names (Bus: name), or Inherit: auto", "text"),
    ]
    nin = ["signs", "Inputs"]

    names: list[str] = []  # set from the names of the incoming lines when the model is compiled

    def init(self):
        self.widths: list[int] = []
        dt = self.text("OutDataTypeStr")
        self.object_names = None
        if dt.lower().startswith("bus:"):
            from .bus import element_names_from_object

            name = dt[4:].strip()
            try:
                obj = self.ctx.evaluate(name)
            except Exception as exc:
                raise BlockError(f"the bus object '{name}' cannot be found: {getattr(exc, 'message', exc)}") from None
            self.object_names = element_names_from_object(obj)
            if len(self.object_names) != self.n_in():
                raise BlockError(
                    f"the bus object '{name}' has {len(self.object_names)} elements but the block has "
                    f"{self.n_in()} inputs"
                )

    def out(self, t, u):
        from .bus import make

        self.widths = [np.size(v) for v in u]
        names = self.object_names or list(self.names) or [f"signal{k + 1}" for k in range(len(u))]
        names = names + [f"signal{k + 1}" for k in range(len(names), len(u))]
        return [make(names[: len(u)], u)]


@block("BusSelector", "Routing", "Picks signals out of a bus by name (nested ones as outer.inner).", icon="bsel")
class BusSelector(Node):
    params = [
        P("OutputSignals", "signal1,signal2", "Signals to pick (names separated by commas)", "text"),
        P("OutputAsBus", "off", "Output the picked signals as one bus", "bool"),
    ]
    nout = ["if", "OutputAsBus", "on", 1, ["csv", "OutputSignals"]]

    def init(self):
        self.names = [s.strip() for s in self.text("OutputSignals").split(",") if s.strip()]

    def out(self, t, u):
        from .bus import make, select, spec_of

        v = u[0]
        if spec_of(v) is None:
            src = self.inputs[0][0] if self.inputs and self.inputs[0] else None
            raise BlockError(
                "the input is not a bus"
                + ("" if src is None else f" (it comes from '{src.path}', which does not make one)")
                + "; feed it from a Bus Creator, a bus, or a struct"
            )
        picked = [select(v, name) for name in self.names]
        if self.is_on("OutputAsBus"):
            return [make([n.split(".")[-1] for n in self.names], picked)]
        return picked


@block(
    "BusAssignment",
    "Routing",
    "Replaces signals of a bus by name with the values at its other inputs.",
    title="Bus Assignment",
    icon="b=",
)
class BusAssignment(Node):
    params = [P("AssignedSignals", "signal1", "Signals to replace (names separated by commas)", "text")]
    nin = ["sum", 1, ["csv", "AssignedSignals"]]

    def init(self):
        self.names = [s.strip() for s in self.text("AssignedSignals").split(",") if s.strip()]

    def out(self, t, u):
        from .bus import assign

        y = u[0]
        for name, v in zip(self.names, u[1:]):
            y = assign(y, name, v)
        return [y]


@block(
    "Merge",
    "Routing",
    "Outputs whichever input was computed most recently (for conditionally run subsystems).",
    icon="mrg",
)
class Merge(Node):
    params = [P("Inputs", "2", "Number of inputs", "text"), P("InitialOutput", "0", "Initial output")]
    nin = ["int", "Inputs"]
    time_varying = True

    def init(self):
        self.last = None
        self.y0 = self.num("InitialOutput")

    def out(self, t, u):
        best, stamp = self.last, -1
        for k, link in enumerate(self.inputs):
            if link is not None and link[0].stamp > stamp:
                best, stamp = k, link[0].stamp
        if stamp < 0 or best is None:
            y0 = self.y0
            if y0.size == 1:  # a scalar initial output takes the size of the signals merged
                big = max((np.asarray(v) for v in u), key=lambda v: v.size, default=y0)
                if big.size > 1:
                    y0 = np.broadcast_to(y0.ravel()[0], big.shape).astype(float)
            return [y0]
        if not self.sim.minor:
            self.last = best
        return [u[best]]


# -- structure-only blocks: the simulator removes them when it flattens the model -------


class Virtual(Node):
    virtual = True


@block(
    "Subsystem",
    "Routing",
    "A group of blocks shown as one block. Double-click to look inside.",
    aliases=("SubSystem",),
    icon="sub",
)
class Subsystem(Virtual):
    params = []
    nin = "system"
    nout = "system"


@block("Goto", "Routing", "Sends its input to every From block with the same tag, without drawing a line.", icon="goto")
class Goto(Virtual):
    params = [P("GotoTag", "A", "Tag", "text")]
    nout = 0


@block("From", "Routing", "Outputs the signal of the Goto block with the same tag.", icon="from")
class From(Virtual):
    params = [P("GotoTag", "A", "Tag", "text")]
    nin = 0


@block(
    "EnablePort",
    "Routing",
    "Put inside a subsystem: the subsystem only runs while its enable input is positive.",
    title="Enable",
    icon="en",
)
class EnablePort(Virtual):
    params = [P("StatesWhenEnabling", "held", "States when enabled again", options=["held", "reset"])]
    nin = 0
    nout = 0


@block(
    "TriggerPort",
    "Routing",
    "Put inside a subsystem: the subsystem runs once each time its trigger input crosses zero.",
    title="Trigger",
    icon="trg",
)
class TriggerPort(Virtual):
    params = [
        P("TriggerType", "rising", "Trigger on", options=["rising", "falling", "either", "function-call"]),
    ]
    nin = 0
    nout = 0


class Gate(Node):
    """Hidden block that decides whether a conditionally run subsystem is active in this step.

    ``kind`` is "enable" (also for action ports), "trigger" or "reset". A reset gate keeps its
    subsystem running and puts the blocks inside back to their initial states when its input fires.
    """

    has_zc = True

    def __init__(self, path, kind, trigger_type, ctx):
        super().__init__(path, {}, ctx)
        self.kind, self.trigger_type = kind, trigger_type
        self.active = kind == "reset"
        self.prev = None
        self.reset_on_enable = False
        self.members: list = []
        self.was_active = False

    def n_in(self):
        return 1

    def n_out(self):
        return 0

    def out(self, t, u):
        if self.sim.minor:
            if self.kind == "trigger":
                self.active = False
            return []
        v = u[0]
        p = self.prev
        up = p is not None and bool(np.any((p <= 0) & (v > 0)))
        down = p is not None and bool(np.any((p > 0) & (v <= 0)))
        if self.kind == "reset":
            how = self.trigger_type.replace(" ", "")
            if how in ("rising", "falling", "either"):
                fire = up if how == "rising" else (down if how == "falling" else up or down)
            else:  # level: every step in which the input is not zero
                fire = bool(np.any(v != 0))
            if fire and not self.sim.initializing:
                self.sim.reset_members(self)
            return []
        if self.kind != "trigger":
            self.active = bool(np.any(v > 0))
            if not self.sim.initializing:
                if self.active and not self.was_active and self.reset_on_enable:
                    self.sim.reset_members(self)
                if not self.active and self.was_active:
                    for n in self.members:
                        if isinstance(n, HeldOutput) and n.reset_when_off:
                            n.y = [n.initial_value()]
            self.was_active = self.active
        else:
            self.active = (
                up if self.trigger_type == "rising" else (down if self.trigger_type == "falling" else up or down)
            )
        return []

    def update(self, t, u):
        self.prev = np.array(u[0], copy=True)
        if self.kind == "trigger":
            self.active = False

    def zc(self, t, u):
        return list(np.ravel(u[0]))


class HeldOutput(Node):
    same_shape = True

    """Output port of a conditionally run subsystem: keeps its value while the subsystem is idle."""

    def __init__(self, path, raw, ctx):
        super().__init__(path, raw, ctx)
        self.initial = raw.get("InitialOutput", "[]")
        self.reset_when_off = str(raw.get("OutputWhenDisabled", "held")).strip().lower() == "reset"

    def n_in(self):
        return 1

    def n_out(self):
        return 1

    def initial_value(self):
        shape = np.shape(self.y[0]) if self.y else (1,)
        if self.initial.strip() in ("", "[]"):
            return np.zeros(shape)
        try:
            return np.array(np.broadcast_to(sig(self.ctx.evaluate(self.initial)), shape), dtype=float)
        except Exception:
            return np.zeros(shape)

    def out(self, t, u):
        return [u[0]]


# =====================================================================================
# User-defined functions
# =====================================================================================


@block(
    "Fcn",
    "User-defined",
    "A formula in u (the input; u(1), u(2)... for a vector) and t. The result is one number.",
    icon="f(u)",
)
class Fcn(Node):
    params = [P("Expr", "sin(u(1))", "Expression", "text")]
    time_varying = True

    def want_min(self, port):
        idx = [int(k) for k in re.findall(r"\bu\s*[\[(]\s*(\d+)\s*[\])]", self.text("Expr"))]
        return (max(idx),) if idx and max(idx) > 1 else None

    def init(self):
        expr = self.text("Expr")
        expr = re.sub(r"\bu\[(\d+)\]", r"u(\1)", expr).replace("!=", "~=").replace("**", "^")
        expr = re.sub(r"!(?!=)", "~", expr)
        self.fn = self.ctx.compile_expr(expr, ["u", "t"])

    def out(self, t, u):
        try:
            v = sig(self.fn(to_value(np.ravel(u[0])), np.array([[t]])))
        except BlockError:
            raise
        except Exception as exc:
            raise BlockError(f"the expression '{self.text('Expr')}' failed: {getattr(exc, 'message', exc)}") from None
        return [v]


_DEFAULT_SCRIPT = "function y = fcn(u)\ny = u;\n"


@block(
    "MATLABFunction",
    "User-defined",
    "Runs a function you write in the MATLAB language; its inputs and outputs become ports.",
    title="MATLAB Function",
    icon="fcn",
)
class MATLABFunction(Node):
    params = [P("Script", _DEFAULT_SCRIPT, "Function code", "code")]
    nin = ["sig", "Script", "in"]
    nout = ["sig", "Script", "out"]
    time_varying = True

    def init(self):
        self.ins, self.outs = signature(self.text("Script"))
        self.call = self.ctx.compile_function(self.text("Script"))

    def out(self, t, u):
        from ..values import Struct
        from .bus import from_struct, spec_of, to_struct

        try:
            args = [to_struct(v) if spec_of(v) is not None else to_value(v) for v in u]
            res = self.call(args, len(self.outs))
            return [from_struct(r) if isinstance(r, Struct) else sig(r) for r in res]
        except BlockError:
            raise
        except Exception as exc:
            line = getattr(exc, "line", None)
            where = f" (line {line} of the block's code)" if line else ""
            raise BlockError(f"{getattr(exc, 'message', exc)}{where}") from None


@block(
    "InterpretedFunction",
    "User-defined",
    "Calls a function handle or a named function with the input: y = f(u).",
    title="Interpreted Function",
    aliases=("MATLABFcn", "InterpretedMATLABFunction", "InterpretedMATLABFcn"),
    icon="@f",
)
class InterpretedFunction(Node):
    params = [
        P("MATLABFcn", "@(u) sin(u)", "Function handle, function name, or expression in u", "text"),
        ONOFF("Output1D", "on", "Treat the result as a 1-D vector"),
    ]
    time_varying = True

    def init(self):
        self.fn = self.ctx.compile_callable(self.text("MATLABFcn"))

    def out(self, t, u):
        try:
            return [sig(self.fn(to_value(u[0])))]
        except BlockError:
            raise
        except Exception as exc:
            raise BlockError(f"'{self.text('MATLABFcn')}' failed: {getattr(exc, 'message', exc)}") from None
