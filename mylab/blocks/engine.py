"""The simulator: turns a model into an ordered list of blocks and steps it through time.

Compilation
    Subsystems, ports, Goto/From pairs are removed ("flattened"), every block
    input is tied to the block output that drives it, sample times are worked
    out, and the blocks are sorted so that a block runs after the blocks whose
    outputs it needs in the same instant (direct feedthrough). A cycle of
    such dependencies is an algebraic loop and is reported as an error.

Stepping
    A *major* step computes all outputs, logs them, and then lets blocks
    update their discrete states. Between major steps an ODE solver
    integrates the continuous states; its trial evaluations are *minor*
    steps and are never logged. The solver always stops exactly at discrete
    sample hits and at the known jump times of sources. Blocks may also give
    zero-crossing values; when one changes sign during a step, the crossing
    time is found by bisection and a major step is taken there.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .library import INF, BlockError, Node, find_type, sig
from .blocks_basic import Display, Inport, Outport, Scope, Sink, ToWorkspace, XYGraph
from .blocks_more import BusCreator, Gate, HeldOutput
from .model import Block, Model, System

ZERO = np.zeros(1)
VARIABLE = {"ode45": "RK45", "ode23": "RK23", "ode113": "DOP853", "ode15s": "BDF", "ode23s": "Radau", "ode23t": "Radau",
            "ode23tb": "Radau", "auto": "RK45", "variablestepauto": "RK45", "variablestepdiscrete": "discrete",
            "daessc": "BDF", "odeN": "RK45"}
# Fixed-step methods as (a, b, c) tableaus.
FIXED = {
    "ode1": ([[]], [1.0], [0.0]),
    "ode2": ([[], [1.0]], [0.5, 0.5], [0.0, 1.0]),
    "ode3": ([[], [0.5], [0.0, 0.75]], [2 / 9, 1 / 3, 4 / 9], [0.0, 0.5, 0.75]),
    "ode4": ([[], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6], [0.0, 0.5, 0.5, 1.0]),
    "ode5": ([[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
              [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]],
             [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0]),
}
FIXED_ALIASES = {"fixedstepauto": "ode3", "fixedstepdiscrete": "discrete", "ode8": "ode5", "ode14x": "ode5", "ode1be": "ode1"}


class SimError(Exception):
    """A simulation or compilation error. ``path`` names the block at fault, if any."""

    def __init__(self, message: str, path: str | None = None, paths: list | None = None, diagnostics: list | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.paths = paths or ([path] if path else [])
        self.diagnostics = diagnostics or []


# =====================================================================================
# Evaluation of parameters and user code
# =====================================================================================

class Context:
    """Evaluates MATLAB expressions for block parameters, in the workspace plus mask variables."""

    def __init__(self, interp: Any = None, frame: Any = None, scope: dict | None = None, shared: dict | None = None):
        if interp is None:
            from ..interpreter import Interpreter
            interp = Interpreter(out=lambda s: None)
        self.interp = interp
        self.frame = frame
        self.scope = scope or {}
        self.shared = shared if shared is not None else {"t0": 0.0, "tf": 10.0, "ext": {}, "cache": {}}

    def child(self, scope: dict) -> "Context":
        merged = dict(self.scope)
        merged.update(scope)
        return Context(self.interp, self.frame, merged, self.shared)

    @property
    def t0(self) -> float:
        return self.shared["t0"]

    @property
    def tf(self) -> float:
        return self.shared["tf"]

    def external_input(self, port: int, t: float) -> np.ndarray:
        fn = self.shared["ext"].get(port)
        return ZERO if fn is None else fn(t)

    def _frames(self, extra: dict | None = None):
        """Workspaces to try, in order: the base workspace, then the caller's (both with mask variables on top)."""
        from ..interpreter import Frame
        interp = self.interp
        seen = []
        for src in (interp.base, self.frame):
            if src is None or any(src is s for s in seen):
                continue
            seen.append(src)
            if not self.scope and not extra:
                yield src
                continue
            f = Frame(interp, src.functions, name="block parameter", file=src.file)
            f.vars = dict(src.vars)
            for name, store in src.globals.items():
                if name in store:
                    f.vars[name] = store[name]
            f.vars.update(self.scope)
            if extra:
                f.vars.update(extra)
            yield f

    def _eval_node(self, node: Any, extra: dict | None = None) -> Any:
        from ..errors import MatlabError
        interp = self.interp
        last: Exception | None = None
        for frame in self._frames(extra):
            interp.frames.append(frame)
            try:
                return interp.eval(node, frame)
            except MatlabError as exc:
                last = exc
            finally:
                interp.frames.pop()
        raise last  # type: ignore[misc]

    @staticmethod
    def _parse_expr(expr: str) -> Any:
        from .. import ast as A
        from ..errors import MatlabError
        from ..parser import parse
        program = parse(expr)
        if len(program.body) != 1 or not isinstance(program.body[0], A.ExprStmt):
            raise MatlabError("a single expression is expected")
        return program.body[0].expr

    def evaluate(self, expr: str) -> Any:
        key = (expr, id(self.scope) if self.scope else 0)
        cache = self.shared["cache"]
        if key not in cache:
            try:  # plain numbers need no interpreter
                cache[key] = np.array([[float(expr)]])
            except ValueError:
                cache[key] = self._eval_node(self._parse_expr(expr))
        return cache[key]

    def compile_expr(self, expr: str, names: list[str]) -> Callable:
        """A function of the named variables that evaluates ``expr`` (other names come from the workspace)."""
        from ..interpreter import Frame
        try:
            node = self._parse_expr(expr)
        except Exception as exc:
            raise BlockError(f"the expression '{expr}' is not valid: {getattr(exc, 'message', exc)}") from None
        interp = self.interp
        src = interp.base
        frame = Frame(interp, src.functions, name="block expression")
        frame.vars = dict(src.vars)
        frame.vars.update(self.scope)

        def call(*values):
            for n, v in zip(names, values):
                frame.vars[n] = v
            interp.frames.append(frame)
            try:
                return interp.eval(node, frame)
            finally:
                interp.frames.pop()
        return call

    def compile_function(self, code: str) -> Callable:
        from ..parser import parse
        try:
            program = parse(code)
        except Exception as exc:
            raise BlockError(f"the function code has a syntax error: {getattr(exc, 'message', exc)}") from None
        if not program.functions:
            raise BlockError("the code must start with a function line, e.g. function y = fcn(u)")
        main = next(iter(program.functions.values()))
        functions = {**self.interp.base.functions, **program.functions}
        interp = self.interp

        def call(args, nout):
            return interp.call_user_function(main, args, nout, functions, None)
        return call

    def compile_callable(self, text: str) -> Callable:
        from ..values import FunctionHandle
        interp = self.interp
        text = text.strip()
        if text.startswith("@"):
            handle = self.evaluate(text)
            return lambda u: interp.call_function_value(handle, [u], 1)[0]
        if text.isidentifier():
            try:
                v = self.evaluate(text)
            except Exception:
                v = None
            if isinstance(v, FunctionHandle):
                return lambda u: interp.call_function_value(v, [u], 1)[0]
            if text != "u" and interp.function_exists(text):
                return lambda u: interp.call_function_value(FunctionHandle(name=text), [u], 1)[0]
        return self.compile_expr(text, ["u"])


# =====================================================================================
# Compilation
# =====================================================================================

class _Sys:
    """One (sub)system while flattening."""

    def __init__(self, system: System, prefix: str, parent: "_Sys | None", block: Block | None, conds: list, ctx: Context):
        self.system, self.prefix, self.parent, self.block, self.conds, self.ctx = system, prefix, parent, block, conds, ctx
        self.gated = False
        self.into = {(ln.dst[0], ln.dst[1]): ln for ln in system.lines}
        self.children: dict[str, _Sys] = {}

    def path(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name


class LogTap(Sink):
    """Hidden sink that records a line marked for logging."""

    def n_in(self):
        return 1


class Compiled:
    def __init__(self, model: Model, ctx: Context):
        self.model = model
        self.ctx = ctx
        self.nodes: list[Node] = []
        self.order: list[Node] = []
        self.diagnostics: list[dict] = []
        self.by_path: dict[str, Node] = {}
        self.taps: dict[str, LogTap] = {}

    def problem(self, level: str, message: str, path: str | None = None, port: int | None = None, kind: str = "") -> None:
        d = {"level": level, "message": message, "path": path, "kind": kind}
        if port is not None:
            d["port"] = port
        self.diagnostics.append(d)

    def errors(self) -> list[dict]:
        return [d for d in self.diagnostics if d["level"] == "error"]


def port_count(block: Block, kind: str) -> int:
    """Number of input ('in') or output ('out') ports of a block as drawn."""
    cls = find_type(block.type)
    if cls is None:
        return 0
    if block.system is not None:
        want = "Inport" if kind == "in" else "Outport"
        return sum(1 for b in block.system.blocks if getattr(find_type(b.type), "type", "") == want)
    params = {p.name: p.default for p in cls.params}
    params.update(block.params)
    from .library import count_ports
    return count_ports(cls.nin if kind == "in" else cls.nout, params)


def _port_number(system: System, block: Block, type_name: str) -> int:
    try:
        return int(float(block.params.get("Port", "")))
    except ValueError:
        same = [b for b in system.blocks if getattr(find_type(b.type), "type", "") == type_name]
        return same.index(block) + 1


def compile_model(model: Model, interp: Any = None, frame: Any = None, strict: bool = True,
                  config: dict | None = None) -> Compiled:
    """Flatten, check and sort a model. With ``strict`` the first error is raised as :class:`SimError`."""
    ctx = Context(interp, frame)
    comp = Compiled(model, ctx)
    cfg = dict(model.config)
    cfg.update(config or {})
    try:
        ctx.shared["t0"] = _number(ctx, cfg.get("StartTime", "0"), "StartTime")
        ctx.shared["tf"] = _number(ctx, cfg.get("StopTime", "10"), "StopTime")
    except SimError as exc:
        comp.problem("error", exc.message)
    node_of: dict[tuple[int, str], Node] = {}
    gotos: dict[str, list[tuple[_Sys, Block]]] = {}
    systems: list[_Sys] = []
    gates: list[tuple[Gate, _Sys, str, str]] = []

    def build(sy: _Sys) -> None:
        systems.append(sy)
        for b in sy.system.blocks:
            cls = find_type(b.type)
            path = sy.path(b.name)
            if cls is None:
                comp.problem("error", f"'{path}': there is no block type '{b.type}' in mylab Blocks.", path, kind="type")
                continue
            kind = cls.type
            if kind == "Subsystem":
                inner = b.system or System()
                conds = list(sy.conds)
                gated = False
                for ib in inner.blocks:
                    it = getattr(find_type(ib.type), "type", "")
                    if it in ("EnablePort", "TriggerPort"):
                        which = "enable" if it == "EnablePort" else "trigger"
                        g = Gate(f"{path}/{ib.name}", which, ib.params.get("TriggerType", "rising").strip().lower(), ctx)
                        g.conds = list(sy.conds)
                        comp.nodes.append(g)
                        gates.append((g, sy, b.name, which))
                        conds.append(g)
                        gated = True
                child_ctx = sy.ctx
                mask = b.extra.get("mask")
                if isinstance(mask, dict) and mask:
                    scope = {}
                    for name, expr in mask.items():
                        try:
                            scope[name] = sy.ctx.child(scope).evaluate(str(expr))
                        except Exception as exc:
                            comp.problem("error", f"'{path}': mask parameter {name} = '{expr}' cannot be evaluated: "
                                         f"{getattr(exc, 'message', exc)}", path, kind="param")
                    child_ctx = sy.ctx.child(scope)
                child = _Sys(inner, path, sy, b, conds, child_ctx)
                child.gated = gated
                sy.children[b.name] = child
                build(child)
                continue
            if kind in ("Goto", "From", "EnablePort", "TriggerPort"):
                if kind == "Goto":
                    gotos.setdefault(b.params.get("GotoTag", "A").strip(), []).append((sy, b))
                continue
            if kind == "Inport" and sy.parent is not None:
                continue
            if kind == "Outport" and sy.parent is not None:
                if not sy.gated:
                    continue
                node: Node = HeldOutput(path, b.params, sy.ctx)
            else:
                node = cls(path, b.params, sy.ctx)
            node.conds = list(sy.conds)
            node.block = b  # type: ignore[attr-defined]
            comp.nodes.append(node)
            comp.by_path[path] = node
            node_of[(id(sy), b.name)] = node

    root = _Sys(model.root, "", None, None, [], ctx)
    build(root)

    def find_goto(sy: _Sys, tag: str):
        found = gotos.get(tag, [])
        s: _Sys | None = sy
        while s is not None:  # nearest enclosing system first
            for gs, gb in found:
                if gs is s:
                    return gs, gb
            s = s.parent
        return found[0] if found else None

    unconnected: set[tuple[str, Any]] = set()

    def driver(sy: _Sys, name: str, port: Any, seen: set) -> tuple[Node, int] | None:
        key = (id(sy), name, port)
        if key in seen:
            return None
        seen.add(key)
        b = sy.system.block(name)
        if b is None or not isinstance(port, int):
            return None
        node = node_of.get((id(sy), name))
        if node is not None and not isinstance(node, HeldOutput):
            return (node, port - 1) if 1 <= port <= max(node.n_out(), 1) and node.n_out() else None
        kind = getattr(find_type(b.type), "type", "")
        if kind == "Subsystem":
            child = sy.children.get(name)
            if child is None:
                return None
            for ob in child.system.blocks:
                if getattr(find_type(ob.type), "type", "") == "Outport" and _port_number(child.system, ob, "Outport") == port:
                    held = node_of.get((id(child), ob.name))
                    return (held, 0) if held is not None else source(child, ob.name, 1, seen)
            return None
        if kind == "Inport" and sy.parent is not None and sy.block is not None:
            return source(sy.parent, sy.block.name, _port_number(sy.system, b, "Inport"), seen)
        if kind == "From":
            hit = find_goto(sy, b.params.get("GotoTag", "A").strip())
            if hit is None:
                comp.problem("error", f"'{sy.path(name)}': there is no Goto block with the tag "
                             f"'{b.params.get('GotoTag', 'A')}'.", sy.path(name), kind="goto")
                return None
            return source(hit[0], hit[1].name, 1, seen)
        return None

    def source(sy: _Sys, name: str, port: Any, seen: set | None = None) -> tuple[Node, int] | None:
        ln = sy.into.get((name, port))
        if ln is None:
            unconnected.add((sy.path(name), port))
            return None
        return driver(sy, ln.src[0], ln.src[1], seen if seen is not None else set())

    # -- wire the inputs --------------------------------------------------------------
    for sy in systems:
        for b in sy.system.blocks:
            node = node_of.get((id(sy), b.name))
            if node is None:
                continue
            n = node.n_in()
            node.inputs = [source(sy, b.name, k + 1) for k in range(n)]
            names = []
            for k in range(n):
                ln = sy.into.get((b.name, k + 1))
                names.append(ln.name if ln is not None and ln.name else "")
            node.in_names = names  # type: ignore[attr-defined]
            if isinstance(node, BusCreator):
                node.names = [nm or f"signal{k + 1}" for k, nm in enumerate(names)]
        for ln in sy.system.lines:
            if ln.log and ln.name:
                tap = LogTap(sy.path(ln.name), {}, sy.ctx)
                tap.inputs = [driver(sy, ln.src[0], ln.src[1], set())]
                tap.conds = list(sy.conds)
                comp.nodes.append(tap)
                comp.taps[ln.name] = tap
    for g, sy, name, which in gates:
        g.inputs = [source(sy, name, which)]

    # -- structural checks -------------------------------------------------------------
    for sy in systems:
        names = {b.name for b in sy.system.blocks}
        outs: dict[tuple, int] = {}
        seen_dst: set = set()
        for ln in sy.system.lines:
            for end, what in ((ln.src, "starts"), (ln.dst, "ends")):
                if end[0] not in names:
                    comp.problem("error", f"A line {what} at '{sy.path(end[0])}', which does not exist.", sy.prefix or None,
                                 kind="line")
            if ln.dst in seen_dst:
                comp.problem("error", f"Input {ln.dst[1]} of '{sy.path(ln.dst[0])}' has more than one line going into it.",
                             sy.path(ln.dst[0]), kind="line")
            seen_dst.add(ln.dst)
            outs[ln.src] = outs.get(ln.src, 0) + 1
            sb, db = sy.system.block(ln.src[0]), sy.system.block(ln.dst[0])
            if sb is not None and isinstance(ln.src[1], int) and ln.src[1] > port_count(sb, "out") and find_type(sb.type):
                comp.problem("error", f"A line starts at output {ln.src[1]} of '{sy.path(sb.name)}', "
                             f"which has {port_count(sb, 'out')} output(s).", sy.path(sb.name), kind="line")
            if db is not None and isinstance(ln.dst[1], int) and ln.dst[1] > port_count(db, "in") and find_type(db.type):
                comp.problem("error", f"A line ends at input {ln.dst[1]} of '{sy.path(db.name)}', "
                             f"which has {port_count(db, 'in')} input(s).", sy.path(db.name), kind="line")
        for b in sy.system.blocks:
            if find_type(b.type) is None:
                continue
            for k in range(port_count(b, "in")):
                if (b.name, k + 1) not in sy.into:
                    unconnected.add((sy.path(b.name), k + 1))
            child = sy.children.get(b.name)
            if child is not None and child.gated:
                for which in ("enable", "trigger"):
                    has = any(getattr(find_type(ib.type), "type", "") == ("EnablePort" if which == "enable" else "TriggerPort")
                              for ib in child.system.blocks)
                    if has and (b.name, which) not in sy.into:
                        unconnected.add((sy.path(b.name), which))
            for k in range(port_count(b, "out")):
                if (b.name, k + 1) not in outs:
                    comp.problem("info", f"Output {k + 1} of '{sy.path(b.name)}' is not connected.", sy.path(b.name), k + 1,
                                 "unconnected-output")
    for path, port in sorted(unconnected, key=str):
        comp.problem("warning", f"Input {port} of '{path}' is not connected; it is read as zero.", path,
                     port if isinstance(port, int) else None, "unconnected-input")

    # -- parameters --------------------------------------------------------------------
    bad: set[int] = set()
    for node in comp.nodes:
        try:
            node.init()
        except BlockError as exc:
            comp.problem("error", f"'{node.path}': {exc}", node.path, kind="param")
            bad.add(id(node))
        except Exception as exc:  # an unexpected failure in one block should not hide the others
            comp.problem("error", f"'{node.path}': {getattr(exc, 'message', None) or exc}", node.path, kind="param")
            bad.add(id(node))

    if not bad:
        _resolve_rates(comp)
        _sort(comp)
    if strict and comp.errors():
        first = comp.errors()[0]
        raise SimError(first["message"], first.get("path"), first.get("paths"), comp.diagnostics)
    return comp


def _number(ctx: Context, expr: Any, name: str) -> float:
    try:
        v = sig(ctx.evaluate(str(expr)))
        if v.size != 1:
            raise BlockError("a single number is needed")
        return float(v[0])
    except Exception as exc:
        raise SimError(f"The setting {name} = '{expr}' cannot be evaluated: {getattr(exc, 'message', exc)}") from None


def _resolve_rates(comp: Compiled) -> None:
    spec: dict[int, Any] = {}
    for n in comp.nodes:
        try:
            s = n.sample_time()
        except BlockError as exc:
            comp.problem("error", f"'{n.path}': {exc}", n.path, kind="param")
            s = None
        if any(g.kind == "trigger" for g in n.conds) or isinstance(n, Gate):
            s = None if s != "const" else s
        spec[id(n)] = s
    pending = [n for n in comp.nodes if spec[id(n)] == "inherit"]

    def settle(n: Node, partial: bool) -> bool:
        srcs = [spec[id(link[0])] for link in n.inputs if link is not None]
        known = [s for s in srcs if s != "inherit"]
        if any(s is None for s in known):
            spec[id(n)] = None
            return True
        if len(known) < len(srcs) and not (partial and known):
            return False
        timed = [s for s in known if s != "const"]
        if not timed:
            if partial and len(known) < len(srcs):
                return False
            spec[id(n)] = None
        else:
            spec[id(n)] = timed[0] if all(s == timed[0] for s in timed) else None
        return True

    for partial in (False, True):
        progress = True
        while progress and pending:
            progress = False
            for n in list(pending):
                if settle(n, partial):
                    pending.remove(n)
                    progress = True
    for n in pending:
        spec[id(n)] = None
    for n in comp.nodes:
        n.rate = spec[id(n)]


def _sort(comp: Compiled) -> None:
    nodes = comp.nodes
    index = {id(n): i for i, n in enumerate(nodes)}
    succ: list[list[int]] = [[] for _ in nodes]
    indeg = [0] * len(nodes)

    def edge(a: Node, b: Node) -> None:
        succ[index[id(a)]].append(index[id(b)])
        indeg[index[id(b)]] += 1

    for n in nodes:
        for k, link in enumerate(n.inputs):
            if link is not None and n.is_df(k):
                edge(link[0], n)
        for g in n.conds:
            edge(g, n)
    # Tarjan's strongly connected components (iterative): any component with a cycle is an algebraic loop.
    low = [0] * len(nodes)
    num = [-1] * len(nodes)
    on = [False] * len(nodes)
    stack: list[int] = []
    counter = 0
    loops: list[list[int]] = []
    for start in range(len(nodes)):
        if num[start] != -1:
            continue
        work = [(start, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                num[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            recurse = False
            for j in range(i, len(succ[v])):
                w = succ[v][j]
                if num[w] == -1:
                    work.append((v, j + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                if on[w]:
                    low[v] = min(low[v], num[w])
            if recurse:
                continue
            if low[v] == num[v]:
                group = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    group.append(w)
                    if w == v:
                        break
                if len(group) > 1 or v in succ[v]:
                    loops.append(group)
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    for group in loops:
        members = set(group)
        # walk the cycle so the message reads in signal order
        chain = [min(group)]
        while True:
            nxt = [w for w in succ[chain[-1]] if w in members and w not in chain]
            if not nxt:
                break
            chain.append(nxt[0])
        paths = [nodes[i].path for i in chain]
        comp.diagnostics.append({
            "level": "error", "kind": "algebraic-loop", "path": paths[0], "paths": paths,
            "message": "Algebraic loop: " + " -> ".join(f"'{p}'" for p in paths + paths[:1]) +
                       ". Each of these blocks needs the output of the one before it at the same instant, so there is "
                       "no order to compute them in. Break the loop with a block that has memory (Integrator, Unit Delay, "
                       "Memory, Transfer Fcn with more poles than zeros)."})
    if loops:
        return
    # Kahn's algorithm, keeping the model's block order where the dependencies allow it.
    import heapq
    ready = [i for i in range(len(nodes)) if indeg[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        v = heapq.heappop(ready)
        order.append(nodes[v])
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, w)
    comp.order = order


def diagnose(model: Model, interp: Any = None, frame: Any = None) -> list[dict]:
    """All problems found without running the model: for the editor's markers."""
    try:
        return compile_model(model, interp, frame, strict=False).diagnostics
    except SimError as exc:
        return [{"level": "error", "message": exc.message, "path": exc.path, "kind": ""}]


# =====================================================================================
# Simulation
# =====================================================================================

class SimResult:
    def __init__(self):
        self.t = np.zeros(0)
        self.x = np.zeros((0, 0))
        self.y = np.zeros((0, 0))
        self.state_names: list[str] = []
        self.scopes: dict[str, dict] = {}
        self.xy: dict[str, dict] = {}
        self.workspace: dict[str, dict] = {}
        self.logs: dict[str, dict] = {}
        self.displays: dict[str, np.ndarray] = {}
        self.outports: list[dict] = []
        self.warnings: list[str] = []
        self.stopped_by: str | None = None
        self.steps = 0
        self.solver = ""


class Sim:
    def __init__(self, comp: Compiled, config: dict | None = None, on_step: Callable | None = None):
        self.comp = comp
        self.order = comp.order
        self.ctx = comp.ctx
        cfg = dict(comp.model.config)
        cfg.update(config or {})
        self.cfg = cfg
        self.minor = True
        self.logging = False
        self.initializing = True
        self.step_no = 0
        self.hits: set = set()
        self.stop_requested: str | None = None
        self.on_step = on_step
        self.x_dirty = False
        self.xwork = np.zeros(0)
        self.t0, self.tf = self.ctx.t0, self.ctx.tf
        if not (math.isfinite(self.t0) and math.isfinite(self.tf)):
            raise SimError("StartTime and StopTime must be finite numbers. To run until a condition is met, use a "
                           "Stop Simulation block and a StopTime that is large enough.")
        if self.tf < self.t0:
            raise SimError(f"StopTime ({self.tf:g}) is before StartTime ({self.t0:g}).")
        for n in self.order:
            n.sim = self
        self.cont = []
        self.rates: dict[tuple, int] = {}
        self.breakers = [n for n in self.order if type(n).next_break is not Node.next_break]
        self.zc_nodes: list[Node] = []
        self.t_log: list[float] = []
        self.x_log: list[np.ndarray] = []
        self._init_states()

    # -- helpers ----------------------------------------------------------------------
    def request_stop(self, path: str) -> None:
        self.stop_requested = path

    def set_state(self, node: Node, values: np.ndarray) -> None:
        self.xwork[node.xo:node.xo + node.nx] = values
        self.x_dirty = True

    @staticmethod
    def gather(n: Node) -> list:
        u = []
        for link in n.inputs:
            if link is None:
                u.append(ZERO)
            else:
                ys = link[0].y
                u.append(ys[link[1]] if link[1] < len(ys) else ZERO)
        return u

    def _fail(self, n: Node, exc: Exception) -> SimError:
        if isinstance(exc, SimError):
            return exc
        msg = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        return SimError(f"'{n.path}': {msg}", n.path)

    def active(self, n: Node) -> bool:
        for g in n.conds:
            if not g.active:
                return False
        return True

    def eval(self, t: float, x: np.ndarray, minor: bool, logging: bool = False) -> None:
        self.minor, self.logging = minor, logging
        init = self.initializing
        hits = self.hits
        for n in self.order:
            if not init:
                if n.conds and not self.active(n):
                    continue
                r = n.rate
                if r is not None and (minor or r == "const" or r not in hits):
                    continue
            if n.nx:
                n.x = x[n.xo:n.xo + n.nx]
            try:
                n.y = n.out(t, self.gather(n))
            except Exception as exc:
                raise self._fail(n, exc) from None
            if not minor:
                n.stamp = self.step_no

    def deriv(self, t: float, x: np.ndarray) -> np.ndarray:
        self.eval(t, x, True)
        dx = np.zeros(x.size)
        for n in self.cont:
            if n.conds and not self.active(n):
                continue
            n.x = x[n.xo:n.xo + n.nx]
            try:
                d = n.der(t, self.gather(n))
            except Exception as exc:
                raise self._fail(n, exc) from None
            if np.size(d) != n.nx:
                raise SimError(f"'{n.path}': the input has {np.size(d)} elements but the block has {n.nx} states.", n.path)
            dx[n.xo:n.xo + n.nx] = d
        if not np.all(np.isfinite(dx)):
            k = int(np.flatnonzero(~np.isfinite(dx))[0])
            who = next((n for n in self.cont if n.xo <= k < n.xo + n.nx), None)
            raise SimError(f"The derivative of '{who.path if who else '?'}' is not finite at t = {t:g} "
                           "(division by zero, or the solution is growing without bound).", who.path if who else None)
        return dx

    def zc_values(self, t: float, x: np.ndarray) -> np.ndarray:
        g: list[float] = []
        for n in self.zc_nodes:
            if n.conds and not self.active(n):
                g += [0.0] * getattr(n, "zc_len", 0)
                continue
            if n.nx:
                n.x = x[n.xo:n.xo + n.nx]
            try:
                v = n.zc(t, self.gather(n))
            except Exception as exc:
                raise self._fail(n, exc) from None
            n.zc_len = len(v)  # type: ignore[attr-defined]
            g += v
        return np.array(g, dtype=float)

    # -- start-up ---------------------------------------------------------------------
    def _init_states(self) -> None:
        order = self.order
        for n in order:
            n.setup(None)
        x = np.zeros(0)
        for attempt in range(8):
            off = 0
            for n in order:
                n.xo = off  # type: ignore[attr-defined]
                off += n.nx
            x = np.zeros(off)
            for n in order:
                if n.nx:
                    x0 = np.asarray(n.initial_state(), float)  # type: ignore[attr-defined]
                    if x0.size != n.nx:
                        raise SimError(f"'{n.path}': the initial state has the wrong size.", n.path)
                    x[n.xo:n.xo + n.nx] = x0
            self.eval(self.t0, x, True)
            self.eval(self.t0, x, True)  # second pass: inputs that come from blocks later in the order
            changed = False
            for n in order:
                shapes = [None if link is None else np.shape(u) for link, u in zip(n.inputs, self.gather(n))]
                try:
                    if n.setup(shapes):
                        changed = True
                except BlockError as exc:
                    raise self._fail(n, exc) from None
            if not changed:
                break
        else:
            raise SimError("The signal sizes do not settle. Check the initial values of blocks in feedback loops.")
        late = False
        for n in order:
            fn = getattr(n, "late_init", None)
            if fn is not None and n.nx:
                v = fn(self.gather(n))
                if v is not None:
                    x[n.xo:n.xo + n.nx] = v
                    late = True
        if late:
            self.eval(self.t0, x, True)
            self.eval(self.t0, x, True)
        for n in order:
            if isinstance(n, HeldOutput) and n.initial.strip() not in ("", "[]"):
                try:
                    n.y = [np.broadcast_to(sig(n.ctx.evaluate(n.initial)), np.shape(n.y[0])).astype(float)]
                except Exception:
                    pass
        self.x0 = x
        self.cont = [n for n in order if n.nx]
        self.zc_nodes = [n for n in order if n.has_zc]
        for n in order:
            if isinstance(n.rate, tuple):
                self.rates.setdefault(n.rate, 0)
        self.initializing = False

    # -- major step -------------------------------------------------------------------
    def hit_time(self, rate: tuple) -> float:
        return self.t0 + rate[1] + self.rates[rate] * rate[0]

    def major(self, t: float, x: np.ndarray) -> np.ndarray:
        self.step_no += 1
        tol = 1e-9 * max(1.0, abs(t)) + getattr(self, "hit_slack", 0.0)
        self.hits = {r for r in self.rates if self.hit_time(r) <= t + tol}
        x = np.array(x, dtype=float)
        self.xwork = x
        self.x_dirty = False
        self.eval(t, x, False, True)
        self.t_log.append(t)
        if self.keep_states:
            self.x_log.append(x.copy())
        step = self.step_no
        for n in self.order:
            if n.stamp == step:
                if n.nx:
                    n.x = x[n.xo:n.xo + n.nx]
                try:
                    n.update(t, self.gather(n))
                except Exception as exc:
                    raise self._fail(n, exc) from None
        for r in self.hits:
            while self.hit_time(r) <= t + tol:
                self.rates[r] += 1
        self.logging = False
        if self.on_step is not None:
            self.on_step(self, t)
        return x

    def next_bound(self, t: float) -> float:
        b = self.tf
        for r in self.rates:
            b = min(b, self.hit_time(r))
        for n in self.breakers:
            if not n.conds or self.active(n):
                b = min(b, n.next_break(t))
        return b if b > t else self.tf

    # -- main loops -------------------------------------------------------------------
    keep_states = False

    def run(self) -> None:
        cfg = self.cfg
        ctx = self.ctx
        name = str(cfg.get("Solver", "ode45")).strip()
        key = name.lower()
        key = FIXED_ALIASES.get(key, key)
        nx = self.x0.size
        span = self.tf - self.t0

        def setting(k: str, default: float) -> float:
            raw = str(cfg.get(k, "auto")).strip()
            return default if raw.lower() in ("auto", "", "-1") else _number(ctx, raw, k)

        self.solver_name = name
        if key in FIXED or key == "discrete":
            if key == "discrete" and nx:
                raise SimError(f"The solver '{name}' is for models without continuous states, but this model has "
                               f"{nx} (for example in '{self.cont[0].path}'). Choose ode45 or ode4.")
            fastest = min((r[0] for r in self.rates), default=None)
            h = setting("FixedStep", fastest if fastest else (span / 50 if span > 0 else 1.0))
            if h <= 0:
                raise SimError("FixedStep must be positive.")
            self.run_fixed(FIXED.get(key, FIXED["ode1"]), h)
            return
        method = VARIABLE.get(key)
        if method is None:
            raise SimError(f"Unknown solver '{name}'. Variable-step: ode45, ode23, ode113, ode15s, ode23s. "
                           "Fixed-step: ode1 (Euler), ode2, ode3, ode4 (Runge-Kutta), ode5.")
        if method == "discrete" and nx:
            raise SimError(f"The solver '{name}' is for models without continuous states, but this model has {nx}. "
                           "Choose ode45.")
        hmax = setting("MaxStep", span / 50 if span > 0 else 1.0)
        rtol, atol = setting("RelTol", 1e-3), setting("AbsTol", 1e-6)
        if hmax <= 0 or rtol <= 0 or atol <= 0:
            raise SimError("MaxStep, RelTol and AbsTol must be positive.")
        self.run_variable(method, hmax, rtol, atol, str(cfg.get("ZeroCrossings", "on")).strip().lower() != "off")

    def run_fixed(self, tableau: tuple, h: float) -> None:
        a, b, c = tableau
        n_steps = int(math.floor((self.tf - self.t0) / h + 1e-9))
        self.hit_slack = h * 1e-6
        x = self.x0.copy()
        nx = x.size
        for k in range(n_steps + 1):
            t = self.t0 + k * h
            x = self.major(t, x)
            if self.stop_requested or k == n_steps:
                break
            if nx:
                ks: list[np.ndarray] = []
                for i in range(len(b)):
                    xi = x + h * sum((a[i][j] * ks[j] for j in range(len(a[i]))), np.zeros(nx))
                    ks.append(self.deriv(t + c[i] * h, xi))
                x = x + h * sum((b[i] * ks[i] for i in range(len(b))), np.zeros(nx))
                if not np.all(np.isfinite(x)):
                    raise SimError(f"The states are no longer finite at t = {t + h:g}. The step size {h:g} may be too "
                                   "large for this model; try a smaller FixedStep or a variable-step solver.")

    def run_variable(self, method: str, hmax: float, rtol: float, atol: float, use_zc: bool) -> None:
        from scipy import integrate
        cls = {"RK45": integrate.RK45, "RK23": integrate.RK23, "DOP853": integrate.DOP853, "BDF": integrate.BDF,
               "Radau": integrate.Radau}.get(method, integrate.RK45)
        t = self.t0
        x = self.major(t, self.x0.copy())
        nx = x.size
        zc_on = use_zc and bool(self.zc_nodes)
        solver = None
        h_hint = None
        tiny_events = 0
        eps = 1e-12 * max(1.0, abs(self.tf), abs(self.t0))
        steps = 0
        while t < self.tf - eps and not self.stop_requested:
            steps += 1
            if steps > 2_000_000:
                raise SimError(f"The simulation needs more than two million steps (it reached t = {t:g}). "
                               "Loosen the tolerances, raise MaxStep, or shorten the run.")
            # State of things just after the major step at t: derivative and zero-crossing values.
            f0 = self.deriv(t, x) if nx else None
            g0 = self.zc_values(t, x) if zc_on else None
            if nx == 0 and zc_on:
                self.eval(t, x, True)
                g0 = self.zc_values(t, x)
            bound = self.next_bound(t)
            if nx == 0:
                t1 = min(t + hmax, bound)
                x1 = x
                dense = None
            else:
                stale = solver is None or solver.t != t or solver.t_bound != bound
                if not stale and hasattr(solver, "f") and not np.allclose(solver.f, f0, rtol=1e-10, atol=1e-13):
                    stale = True
                if stale:
                    first = None if h_hint is None else max(min(h_hint, bound - t), 1e-14)
                    try:
                        solver = cls(self.deriv, t, x, bound, max_step=hmax, rtol=rtol, atol=atol, first_step=first)
                    except ValueError:
                        solver = cls(self.deriv, t, x, bound, max_step=hmax, rtol=rtol, atol=atol)
                message = solver.step()
                if solver.status == "failed":
                    raise SimError(f"The solver could not continue at t = {solver.t:g}: {message or 'step size too small'}. "
                                   "The model may be stiff (try ode15s) or have a discontinuity without a zero crossing.")
                t1, x1 = float(solver.t), np.array(solver.y)
                dense = solver.dense_output()
                h_hint = getattr(solver, "h_abs", None)
            event = False
            if zc_on and g0 is not None and g0.size:
                self.eval(t1, x1, True)
                g1 = self.zc_values(t1, x1)
                if g1.size == g0.size and np.any(g0 * g1 < 0):
                    t1, x1 = self.locate(t, t1, x1, g0, dense)
                    event = True
                    tiny_events = tiny_events + 1 if t1 - t <= 1e-9 * max(1.0, abs(t)) else 0
                    if tiny_events > 500:
                        raise SimError(f"More than 500 zero crossings in a row near t = {t1:g}. A signal is switching "
                                       "back and forth infinitely fast (chattering). Add hysteresis (for example a Relay "
                                       "with different switch-on and switch-off points) or stop the simulation earlier.")
            x_new = self.major(t1, x1)
            if event or self.x_dirty or nx == 0:
                solver = None
            t, x = t1, x_new

    def locate(self, ta: float, tb: float, xb: np.ndarray, g0: np.ndarray, dense: Any) -> tuple[float, np.ndarray]:
        """Bisect for the first sign change of a zero-crossing value in (ta, tb]; returns a point just after it."""
        lo, hi, xhi = ta, tb, xb
        tol = 1e-10 * max(1.0, abs(tb))
        for _ in range(60):
            if hi - lo <= tol:
                break
            mid = 0.5 * (lo + hi)
            xm = np.array(dense(mid)) if dense is not None else xb
            self.eval(mid, xm, True)
            gm = self.zc_values(mid, xm)
            if gm.size == g0.size and np.any(g0 * gm < 0):
                hi, xhi = mid, xm
            else:
                lo = mid
        return hi, xhi

    # -- results ----------------------------------------------------------------------
    def result(self) -> SimResult:
        res = SimResult()
        res.t = np.array(self.t_log)
        res.steps = len(self.t_log)
        res.solver = getattr(self, "solver_name", "")
        res.stopped_by = self.stop_requested
        if self.keep_states and self.x_log:
            res.x = np.array(self.x_log)
        for n in self.cont:
            res.state_names += [n.path] * n.nx
        outs = []
        for n in self.order:
            if isinstance(n, LogTap):
                continue
            if isinstance(n, Scope):
                names = getattr(n, "in_names", [])
                res.scopes[n.path] = {"t": np.array(n.log_t), "title": n.text("Title"),
                                      "signals": [{"name": names[k] if k < len(names) else "", "data": n.data(k)}
                                                  for k in range(len(n.log_u))]}
            elif isinstance(n, XYGraph):
                res.xy[n.path] = {"t": np.array(n.log_t), "x": n.data(0), "y": n.data(1)}
            elif isinstance(n, ToWorkspace):
                res.workspace[n.var] = {"t": np.array(n.log_t), "data": n.data(0), "format": n.format, "path": n.path}
            elif isinstance(n, Display):
                res.displays[n.path] = getattr(n, "last", np.zeros(0))
            elif isinstance(n, Outport):
                outs.append((int(float(n.text("Port") or 1)), n))
        outs.sort(key=lambda p: p[0])
        cols = []
        for port, n in outs:
            data = n.data(0)
            # an output that runs at a slower rate is held between its samples
            if len(n.log_t) != res.t.size:
                idx = np.clip(np.searchsorted(np.array(n.log_t), res.t, side="right") - 1, 0, max(len(n.log_t) - 1, 0))
                data = data[idx] if len(n.log_t) else np.zeros((res.t.size, 1))
            res.outports.append({"path": n.path, "port": port, "t": np.array(n.log_t), "data": n.data(0)})
            cols.append(data)
        res.y = np.hstack(cols) if cols else np.zeros((res.t.size, 0))
        for name, tap in self.comp.taps.items():
            res.logs[name] = {"t": np.array(tap.log_t), "data": tap.data(0)}
        res.warnings = [d["message"] for d in self.comp.diagnostics if d["level"] == "warning"]
        return res


def _external(spec: Any, n_ports: int) -> dict:
    """Input functions from a matrix [t, u1, u2, ...] given to sim as ExternalInput."""
    m = np.asarray(spec, float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise SimError("ExternalInput must be a matrix with time in the first column and one column per input.")
    t, d = m[:, 0], m[:, 1:]
    fns = {}
    for k in range(d.shape[1]):
        fns[k + 1] = (lambda tq, _c=d[:, k]: np.array([np.interp(tq, t, _c)]))
    return fns


def simulate(model: Model, interp: Any = None, frame: Any = None, config: dict | None = None,
             on_step: Callable | None = None, external_input: Any = None, keep_states: bool = False) -> SimResult:
    """Compile and run a model; returns the logged data."""
    comp = compile_model(model, interp, frame, strict=True, config=config)
    if external_input is not None:
        comp.ctx.shared["ext"] = _external(external_input, 0)
    sim = Sim(comp, config, on_step)
    sim.keep_states = keep_states
    sim.run()
    return sim.result()


# =====================================================================================
# Linearization and trimming
# =====================================================================================

class _Plant:
    """dx/dt = f(x, u), y = g(x, u) of a model at its start time, for linearize and trim."""

    def __init__(self, model: Model, interp: Any, frame: Any):
        self.comp = compile_model(model, interp, frame, strict=True)
        self.sim = Sim(self.comp)
        nodes = self.comp.order
        self.ins = sorted((n for n in nodes if isinstance(n, Inport)), key=lambda n: n.port)
        self.outs = sorted((n for n in nodes if isinstance(n, Outport)), key=lambda n: int(float(n.text("Port") or 1)))
        self.nx = self.sim.x0.size
        self.nu = len(self.ins)
        self.u = np.zeros(self.nu)
        ext = self.comp.ctx.shared["ext"]
        for k, n in enumerate(self.ins):
            ext[n.port] = (lambda t, _k=k: self.u[_k:_k + 1])
        self.t = self.sim.t0
        self.sim.initializing = True  # evaluate every block, whatever its sample time
        self.ny = sum(np.size(self.sim.gather(n)[0]) for n in self.outs) if self.outs else 0

    def __call__(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.u = np.asarray(u, float)
        dx = self.sim.deriv(self.t, np.asarray(x, float))
        y = np.concatenate([np.ravel(self.sim.gather(n)[0]) for n in self.outs]) if self.outs else np.zeros(0)
        return dx, y


def linearize(model: Model, interp: Any = None, frame: Any = None, x0: Any = None, u0: Any = None) -> dict:
    """Matrices A, B, C, D of the model linearized around a point, by central differences.

    The inputs are the top-level Inport blocks and the outputs the top-level
    Outport blocks. Discrete states are held at their initial values.
    """
    plant = _Plant(model, interp, frame)
    x = plant.sim.x0.copy() if x0 is None else np.ravel(np.asarray(x0, float))
    u = np.zeros(plant.nu) if u0 is None else np.ravel(np.asarray(u0, float))
    if x.size != plant.nx or u.size != plant.nu:
        raise SimError(f"The model has {plant.nx} continuous states and {plant.nu} inputs, but the operating point has "
                       f"{x.size} states and {u.size} inputs.")
    nx, nu = plant.nx, plant.nu
    _, y0 = plant(x, u)
    ny = y0.size
    A, B, C, D = np.zeros((nx, nx)), np.zeros((nx, nu)), np.zeros((ny, nx)), np.zeros((ny, nu))
    for k in range(nx):
        h = 1e-5 * (1.0 + abs(x[k]))
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        (fp, gp), (fm, gm) = plant(xp, u), plant(xm, u)
        A[:, k], C[:, k] = (fp - fm) / (2 * h), (gp - gm) / (2 * h)
    for k in range(nu):
        h = 1e-5 * (1.0 + abs(u[k]))
        up, um = u.copy(), u.copy()
        up[k] += h
        um[k] -= h
        (fp, gp), (fm, gm) = plant(x, up), plant(x, um)
        B[:, k], D[:, k] = (fp - fm) / (2 * h), (gp - gm) / (2 * h)
    clean = [np.where(np.abs(m) < 1e-9, 0.0, m) for m in (A, B, C, D)]
    names = []
    for n in plant.sim.cont:
        names += [n.path] * n.nx
    return {"A": clean[0], "B": clean[1], "C": clean[2], "D": clean[3], "states": names, "x": x, "u": u, "y": y0}


def trim(model: Model, interp: Any = None, frame: Any = None, x0: Any = None, u0: Any = None) -> dict:
    """A steady state: states x and inputs u for which all derivatives are zero (least squares from a guess)."""
    from scipy import optimize
    plant = _Plant(model, interp, frame)
    x = plant.sim.x0.copy() if x0 is None else np.ravel(np.asarray(x0, float))
    u = np.zeros(plant.nu) if u0 is None else np.ravel(np.asarray(u0, float))
    if x.size != plant.nx or u.size != plant.nu:
        raise SimError(f"The model has {plant.nx} continuous states and {plant.nu} inputs, but the guess has "
                       f"{x.size} states and {u.size} inputs.")
    nx = plant.nx
    if nx == 0:
        _, y = plant(x, u)
        return {"x": x, "u": u, "y": y, "dx": np.zeros(0)}
    sol = optimize.least_squares(lambda z: plant(z[:nx], z[nx:])[0], np.concatenate([x, u]), xtol=1e-12, ftol=1e-12)
    xs, us = sol.x[:nx], sol.x[nx:]
    dx, y = plant(xs, us)
    return {"x": xs, "u": us, "y": y, "dx": dx}
