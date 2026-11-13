"""Signal routing, subsystem ports and user-defined function blocks."""

from __future__ import annotations

import re

import numpy as np

from .library import BlockError, Node, ONOFF, P, block, sig, signature, to_value


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


class _Indexed(Node):
    """Shared index handling of Selector and Assignment."""

    def read_indices(self):
        self.ndim = int(self.scalar("NumberOfDimensions"))
        if self.ndim not in (1, 2):
            raise BlockError("only 1 and 2 dimensions are supported")
        opts = [o.lower() for o in _split_top(self.text("IndexOptions"))]
        exprs = _split_top(self.text("Indices"))
        self.port_index = any("port" in o for o in opts)
        if self.port_index and self.ndim != 1:
            raise BlockError("an index from an input port is only supported for 1 dimension")
        self.idx = []
        for d in range(self.ndim):
            opt = opts[d] if d < len(opts) else "index vector (dialog)"
            if "all" in opt:
                self.idx.append(None)
            elif "port" in opt:
                self.idx.append("port")
            elif "dialog" in opt:
                expr = exprs[d] if d < len(exprs) else "1"
                try:
                    v = sig(self.ctx.evaluate(expr))
                except Exception as exc:
                    raise BlockError(
                        f"the index '{expr}' cannot be evaluated: {getattr(exc, 'message', exc)}"
                    ) from None
                if "starting" in opt:
                    raise BlockError("the 'Starting index' options are not supported; use an index vector")
                self.idx.append(v.astype(int) - (0 if self.zero_based else 1))
            else:
                raise BlockError(f"the index option '{opt}' is not supported")

    @property
    def zero_based(self):
        return self.choice("IndexMode") == "zerobased"

    def key(self, a: np.ndarray, port_value=None):
        if self.ndim == 1:
            i = self.idx[0]
            if isinstance(i, str):
                i = np.round(np.ravel(port_value)).astype(int) - (0 if self.zero_based else 1)
            if i is not None and i.size and (i.min() < 0 or i.max() >= a.size):
                raise BlockError(f"an index is outside the input, which has {a.size} elements")
            return slice(None) if i is None else i
        if a.ndim != 2:
            a = a.reshape(-1, 1)
        rows = np.arange(a.shape[0]) if self.idx[0] is None else self.idx[0]
        cols = np.arange(a.shape[1]) if self.idx[1] is None else self.idx[1]
        if (
            rows.size
            and (rows.min() < 0 or rows.max() >= a.shape[0])
            or cols.size
            and (cols.min() < 0 or cols.max() >= a.shape[1])
        ):
            raise BlockError(f"an index is outside the input, which is {a.shape[0]}x{a.shape[1]}")
        return np.ix_(rows, cols)


_INDEX_PARAMS = [
    P("NumberOfDimensions", "1", "Number of dimensions", options=["1", "2"]),
    P("IndexMode", "One-based", "Index numbering", options=["One-based", "Zero-based"]),
    P(
        "IndexOptions",
        "Index vector (dialog)",
        "Index option per dimension (Select all, Index vector (dialog), Index vector (port))",
        "text",
    ),
    P("Indices", "[1 3]", "Indices, one expression per dimension", "text"),
]


@block("Selector", "Routing", "Picks elements of a vector (or rows and columns of a matrix) by index.", icon="u(i)")
class Selector(_Indexed):
    params = _INDEX_PARAMS + [P("InputPortWidth", "-1", "Width of the input, if known (-1: any)")]

    def want_in(self, port):
        if port or self.ndim != 1:
            return None
        try:
            n = int(self.scalar("InputPortWidth"))
        except BlockError:
            return None
        return (n,) if n > 1 else None

    def want_min(self, port):
        i = self.idx[0] if self.ndim == 1 else None
        return (int(i.max()) + 1,) if isinstance(i, np.ndarray) and i.size and i.max() > 0 else None

    nin = ["if", "IndexOptions", "Index vector (port)", 2, 1]

    def init(self):
        self.read_indices()

    def out(self, t, u):
        a = u[0]
        if self.ndim == 1:
            return [np.ravel(a, order="F")[self.key(np.ravel(a), u[1] if self.port_index else None)]]
        a2 = a if a.ndim == 2 else a.reshape(-1, 1)
        return [sig(a2[self.key(a2)])]


@block("Assignment", "Routing", "Replaces chosen elements of the first input with the second input.", icon="y(i)=u")
class Assignment(_Indexed):
    params = _INDEX_PARAMS
    nin = ["if", "IndexOptions", "Index vector (port)", 3, 2]

    def init(self):
        self.read_indices()

    def out(self, t, u):
        if self.ndim == 1:
            y = np.ravel(u[0], order="F").astype(float).copy()
            y[self.key(y, u[2] if self.port_index else None)] = np.ravel(u[1], order="F")
            return [y]
        y = (u[0] if u[0].ndim == 2 else u[0].reshape(-1, 1)).copy()
        key = self.key(y)
        y[key] = np.reshape(u[1], y[key].shape, order="F") if u[1].size > 1 else u[1]
        return [sig(y)]


@block(
    "BusCreator", "Routing", "Groups signals under their names so a Bus Selector can pick them out again.", icon="bus"
)
class BusCreator(Node):
    params = [P("Inputs", "2", "Number of inputs", "text")]
    nin = ["signs", "Inputs"]

    names: list[str] = []  # set from the names of the incoming lines when the model is compiled

    def init(self):
        self.widths: list[int] = []

    def out(self, t, u):
        self.widths = [v.size for v in u]
        return [np.concatenate([np.ravel(v, order="F") for v in u]) if u else np.zeros(0)]


@block("BusSelector", "Routing", "Picks signals by name out of the bus made by a Bus Creator.", icon="bsel")
class BusSelector(Node):
    params = [P("OutputSignals", "signal1,signal2", "Signals to pick (names separated by commas)", "text")]
    nout = ["csv", "OutputSignals"]

    def init(self):
        self.names = [s.strip() for s in self.text("OutputSignals").split(",") if s.strip()]

    def out(self, t, u):
        src = self.inputs[0][0] if self.inputs and self.inputs[0] else None
        if not isinstance(src, BusCreator):
            raise BlockError("a Bus Selector must be fed by a Bus Creator")
        v, res = u[0], []
        for name in self.names:
            if name not in src.names:
                raise BlockError(f"the bus has no signal '{name}'; it has: {', '.join(src.names)}")
            k = src.names.index(name)
            start = sum(src.widths[:k])
            res.append(v[start : start + src.widths[k]])
        return res


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
    params = []
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
    params = [P("TriggerType", "rising", "Trigger on", options=["rising", "falling", "either"])]
    nin = 0
    nout = 0


class Gate(Node):
    """Hidden block that decides whether a conditionally run subsystem is active in this step."""

    has_zc = True

    def __init__(self, path, kind, trigger_type, ctx):
        super().__init__(path, {}, ctx)
        self.kind, self.trigger_type = kind, trigger_type
        self.active = False
        self.prev = None

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
        if self.kind != "trigger":
            self.active = bool(np.any(v > 0))
        else:
            p = self.prev
            up = p is not None and bool(np.any((p <= 0) & (v > 0)))
            down = p is not None and bool(np.any((p > 0) & (v <= 0)))
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

    def n_in(self):
        return 1

    def n_out(self):
        return 1

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
        try:
            res = self.call([to_value(v) for v in u], len(self.outs))
            return [sig(r) for r in res]
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
