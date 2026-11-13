"""More blocks: if/else action signals, larger lookup tables and a few dynamic limiters."""

from __future__ import annotations

import numpy as np

from .blocks_more import _split_top
from .library import BlockError, Node, P, block, sig


@block(
    "If",
    "Logic",
    "Tests conditions on its inputs (u1, u2...) and switches on the action subsystem of the first one that holds.",
    icon="if",
)
class If(Node):
    params = [
        P("NumInputs", "1", "Number of inputs", "text"),
        P("IfExpression", "u1 > 0", "If condition, in u1, u2...", "text"),
        P("ElseIfExpressions", "", "Else-if conditions, separated by commas", "text"),
        P("ShowElse", "on", "Else output", "bool"),
    ]
    nin = ["int", "NumInputs"]
    nout = ["sum", 1, ["csv0", "ElseIfExpressions"], ["if", "ShowElse", "on", 1, 0]]
    time_varying = True

    def init(self):
        n = self.n_in()
        names = [f"u{k + 1}" for k in range(n)]
        exprs = [self.text("IfExpression")] + [e for e in _split_top(self.text("ElseIfExpressions")) if e]
        self.tests = [self.ctx.compile_expr(e, names) for e in exprs]
        self.has_else = self.is_on("ShowElse")
        self.choice_now = None

    def out(self, t, u):
        from ..values import is_true

        if self.sim.minor and self.choice_now is not None:
            pick = self.choice_now
        else:
            pick = len(self.tests)
            args = [v.reshape(-1, 1) if v.size > 1 else v.reshape(1, 1) for v in u]
            for k, test in enumerate(self.tests):
                try:
                    hit = is_true(test(*args))
                except Exception as exc:
                    raise BlockError(f"a condition could not be evaluated: {getattr(exc, 'message', exc)}") from None
                if hit:
                    pick = k
                    break
            if not self.sim.minor:
                self.choice_now = pick
        n = len(self.tests) + (1 if self.has_else else 0)
        return [np.array([1.0 if k == pick else 0.0]) for k in range(n)]


@block(
    "ActionPort",
    "Routing",
    "Put inside a subsystem: the subsystem runs only while the If block output wired to it is on.",
    title="Action Port",
    icon="act",
)
class ActionPort(Node):
    params = []
    nin = 0
    nout = 0
    virtual = True


def _interp_nd(table: np.ndarray, grids: list[np.ndarray], point: list[float], clip: bool) -> float:
    """Multilinear interpolation of one point; outside the grid the edge cell is extended (or clipped)."""
    idx, frac = [], []
    for g, x in zip(grids, point):
        if clip:
            x = min(max(x, g[0]), g[-1])
        i = int(np.clip(np.searchsorted(g, x, side="right") - 1, 0, g.size - 2))
        idx.append(i)
        frac.append((x - g[i]) / (g[i + 1] - g[i]))
    total = 0.0
    n = len(grids)
    for corner in range(1 << n):
        w, pos = 1.0, []
        for d in range(n):
            hi = (corner >> d) & 1
            w *= frac[d] if hi else 1.0 - frac[d]
            pos.append(idx[d] + hi)
        if w:
            total += w * float(table[tuple(pos)])
    return total


@block(
    "LookupTableND",
    "Nonlinear",
    "Interpolates in a table with three or more dimensions; one input per dimension.",
    title="n-D Lookup Table",
    icon="lutn",
)
class LookupTableND(Node):
    params = [
        P("NumberOfTableDimensions", "3", "Number of dimensions", "text"),
        P("Table", "reshape(1:27, 3, 3, 3)", "Table"),
        P("Breakpoints", "[1 2 3], [1 2 3], [1 2 3]", "Breakpoints, one list per dimension", "text"),
        P("ExtrapMethod", "Linear", "Outside the table", options=["Linear", "Clip"]),
    ]
    nin = ["int", "NumberOfTableDimensions"]

    def init(self):
        n = self.n_in()
        if n < 1:
            raise BlockError("the number of dimensions must be 1 or more")
        self.table = np.asarray(self.value("Table"), dtype=float)
        parts = _split_top(self.text("Breakpoints"))
        if len(parts) != n:
            raise BlockError(f"{n} lists of breakpoints are needed, one per dimension; {len(parts)} were given")
        self.grids = []
        for k, expr in enumerate(parts):
            try:
                g = sig(self.ctx.evaluate(expr))
            except BlockError:
                raise
            except Exception as exc:
                raise BlockError(
                    f"breakpoints {k + 1} = '{expr}' cannot be evaluated: {getattr(exc, 'message', exc)}"
                ) from None
            if g.size < 2 or np.any(np.diff(g) <= 0):
                raise BlockError(f"breakpoints {k + 1} must be 2 or more increasing numbers")
            self.grids.append(g)
        shape = tuple(g.size for g in self.grids)
        if self.table.size != int(np.prod(shape)):
            raise BlockError(f"the table has {self.table.size} values but the breakpoints call for {shape}")
        self.table = self.table.reshape(shape) if self.table.shape != shape else self.table
        self.clip = self.choice("ExtrapMethod") == "clip"

    def out(self, t, u):
        us = np.broadcast_arrays(*[np.ravel(v) for v in u])
        out = [_interp_nd(self.table, self.grids, [float(a[k]) for a in us], self.clip) for k in range(us[0].size)]
        return [np.array(out)]


@block(
    "DirectLookupTable",
    "Nonlinear",
    "Picks an element (or a column) of a table by index, counting from 0; no interpolation.",
    title="Direct Lookup Table",
    aliases=("LookupNDDirect",),
    icon="lut[i]",
)
class DirectLookupTable(Node):
    params = [
        P("NumberOfTableDimensions", "2", "Number of table dimensions", "text"),
        P(
            "InputsSelectThisObjectFromTable",
            "Element",
            "The inputs select",
            options=["Element", "Vector", "2-D Matrix"],
        ),
        P("Table", "[4 5 6;16 19 20;10 18 23]", "Table"),
    ]
    nin = ["dlt", "NumberOfTableDimensions", "InputsSelectThisObjectFromTable"]

    def init(self):
        self.table = np.asarray(self.value("Table"), dtype=float)
        nd = int(self.scalar("NumberOfTableDimensions"))
        if nd == 1:
            self.table = self.table.reshape(-1)
        elif self.table.ndim < nd:
            self.table = self.table.reshape(self.table.shape + (1,) * (nd - self.table.ndim))
        self.keep = {"element": 0, "vector": 1, "2dmatrix": 2}.get(self.choice("InputsSelectThisObjectFromTable"), 0)
        if self.n_in() != self.table.ndim - self.keep and not (nd == 1 and self.keep == 1):
            raise BlockError("the table does not have the number of dimensions given")

    def out(self, t, u):
        tab = self.table
        if not u:  # a 1-D table read as a whole vector
            return [tab.copy()]
        idx = []
        for k, v in enumerate(u):
            i = int(np.floor(float(v.flat[0]) + 0.5))
            idx.append(min(max(i, 0), tab.shape[self.keep + k] - 1))  # out-of-range indices use the edge
        return [sig(tab[(slice(None),) * self.keep + tuple(idx)])]


@block(
    "Prelookup",
    "Nonlinear",
    "Finds where the input lies in a list of breakpoints: the interval number (from 0) and the fraction inside it.",
    aliases=("PreLookup",),
    icon="k,f",
)
class Prelookup(Node):
    params = [
        P("BreakpointsData", "[10:10:110]", "Breakpoints"),
        P("OutputSelection", "Index and fraction", "Outputs", options=["Index and fraction", "Index only"]),
    ]
    nout = ["if", "OutputSelection", "Index only", 1, 2]

    def init(self):
        self.g = self.num("BreakpointsData")
        if self.g.size < 2 or np.any(np.diff(self.g) <= 0):
            raise BlockError("the breakpoints must be 2 or more increasing numbers")
        self.index_only = self.choice("OutputSelection") == "indexonly"

    def out(self, t, u):
        g = self.g
        x = np.clip(u[0], g[0], g[-1])
        last = g.size - 1 if self.index_only else g.size - 2
        k = np.clip(np.searchsorted(g, x, side="right") - 1, 0, last)
        if self.index_only:
            return [k.astype(float)]
        return [k.astype(float), (x - g[k]) / (g[k + 1] - g[k])]


@block(
    "RateLimiterDynamic",
    "Nonlinear",
    "Limits how fast the output may change; the rising and falling limits are inputs (up, u, lo).",
    title="Rate Limiter Dynamic",
    icon="rate",
)
class RateLimiterDynamic(Node):
    params = [P("InitialCondition", "0", "Initial output")]
    nin = 3
    expandable = True

    def init(self):
        self.y_prev = self.num("InitialCondition")
        self.t_prev: float | None = None

    def _y(self, t, u):
        if self.t_prev is None or t <= self.t_prev:
            return self.y_prev + 0.0 * u[1]
        dt = t - self.t_prev
        return self.y_prev + np.clip(u[1] - self.y_prev, u[2] * dt, u[0] * dt)

    def out(self, t, u):
        return [np.asarray(self._y(t, u), float)]

    def update(self, t, u):
        self.y_prev, self.t_prev = np.asarray(self._y(t, u), float), t


@block(
    "MinMaxRunningResettable",
    "Math",
    "Smallest or largest value its first input has had so far; the second input resets it.",
    title="MinMax Running Resettable",
    icon="run",
)
class MinMaxRunningResettable(Node):
    params = [P("Function", "min", "Function", options=["min", "max"]), P("vinit", "0", "Initial value")]
    nin = 2

    def init(self):
        self.best = self.num("vinit")
        self.start = self.best
        self.f = np.minimum if self.choice("Function") == "min" else np.maximum

    def _y(self, u):
        base = self.start if np.any(u[1] != 0) else self.best
        return np.asarray(self.f(base, u[0]), float)

    def out(self, t, u):
        return [self._y(u)]

    def update(self, t, u):
        self.best = self._y(u)


@block(
    "PermuteDimensions",
    "Math",
    "Reorders the dimensions of a matrix signal ([2 1] swaps rows and columns).",
    title="Permute Dimensions",
    icon="perm",
)
class PermuteDimensions(Node):
    params = [P("Order", "[2,1]", "Order of the dimensions")]

    def out(self, t, u):
        order = [int(k) - 1 for k in self.num("Order")]
        a = u[0] if u[0].ndim == 2 else u[0].reshape(-1, 1)
        if sorted(order) != [0, 1]:
            raise BlockError("only two-dimensional signals can be permuted: the order must be [1 2] or [2 1]")
        return [sig(np.transpose(a, order))]
