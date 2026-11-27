"""State charts: the Chart block.

A chart is a hierarchy of states. The children of a state (or of the chart
itself) are either *exclusive* (one of them is active at a time) or
*parallel* (all of them are active, and run one after the other). Transitions
join states and connective junctions; a transition is labelled

    event[condition]{condition action}/transition action

where every part is optional. States are labelled with their actions:

    Name
    entry: y = 1;
    during: n = n + 1;
    exit: y = 0;

(also ``en:``, ``du:``, ``ex:``; lines before the first keyword are entry
actions). The actions and conditions are MATLAB code run by mylab's
interpreter. ``send(E)`` (or a statement that is just the event name)
broadcasts an event; ``after(n, tick)``, ``before``, ``every`` and ``at``,
with ``tick`` or ``sec``, count from the moment the source state was entered;
``in(A.B)`` tests whether a state is active.

How a chart runs (one *step*, at each sample hit or input event):

1. The first step enters the chart: its default transition is followed to a
   state, whose entry action runs, and so on down into its children.
2. Later steps start at the top. For an active state, its outgoing
   transitions are tried in order. The first one whose event matches and
   whose condition holds is taken (through junctions, backtracking when a
   junction has no way on): the source state and its active children are
   exited (exit actions, innermost first), the transition actions run, and
   the destination is entered. If no transition is taken, the state's
   during action runs and its active children are processed the same way.
3. A chart without states is a flow chart: its default transition paths
   run at every step.

The chart description is plain data, stored in the model file under
``"chart"`` (see :func:`ChartDef.from_json`); :class:`ChartBuilder` makes one
from code.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .library import BlockError, Node, P, block, sig, to_value

ACTION_KEYS = {"entry": "entry", "en": "entry", "during": "during", "du": "during", "exit": "exit", "ex": "exit"}
TEMPORAL = re.compile(r"\b(after|before|every|at)\s*\(")
IN_STATE = re.compile(r"\bin\s*\(\s*([A-Za-z_][\w.]*)\s*\)")


# -------------------------------------------------------------------------------------
# Labels
# -------------------------------------------------------------------------------------


def parse_state_label(label: str) -> dict:
    """{'name', 'entry', 'during', 'exit'} from a state label."""
    lines = (label or "").replace("\r", "").split("\n")
    first = lines[0].strip() if lines else ""
    name = first
    rest_first = ""
    body = lines[1:]
    if re.match(r"^(entry|en|during|du|exit|ex)\s*[,:]", first):  # actions only, no name
        name, body = "", lines
    else:
        m = re.match(r"^([A-Za-z_]\w*)\s*/?\s*(.*)$", first)
        if m:
            name, rest_first = m.group(1), m.group(2)
        body = ([rest_first] if rest_first else []) + body
    out = {"name": name, "entry": [], "during": [], "exit": []}
    current = ["entry"]
    for line in body:
        m = re.match(r"^\s*((?:entry|en|during|du|exit|ex)(?:\s*,\s*(?:entry|en|during|du|exit|ex))*)\s*:(.*)$", line)
        if m:
            current = [ACTION_KEYS[k.strip()] for k in m.group(1).split(",")]
            line = m.group(2)
        if line.strip():
            for k in current:
                out[k].append(line)
    return {k: ("\n".join(v) if isinstance(v, list) else v) for k, v in out.items()}


def _matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index of the bracket that closes the one at ``start`` (quotes are skipped)."""
    depth = 0
    i = start
    quote = None
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                quote = None
        elif c in "'\"" and (c == '"' or i == 0 or not (text[i - 1].isalnum() or text[i - 1] in ")]}_.'")):
            quote = c
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise BlockError(f"the label '{text}' has an unclosed '{open_ch}'")


def parse_transition_label(label: str) -> dict:
    """{'events': [...], 'condition', 'conditionAction', 'action'} from a transition label."""
    s = (label or "").strip()
    out = {"events": [], "condition": "", "conditionAction": "", "action": ""}
    i = 0
    # the event part runs to the first [ { or / outside parentheses
    depth = 0
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and c in "[{/":
            break
        i += 1
    trig = s[:i].strip()
    if trig:
        out["events"] = [e.strip() for e in re.split(r"\|(?![|])", trig) if e.strip()]
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c == "[":
            j = _matching(s, i, "[", "]")
            out["condition"] = s[i + 1 : j].strip()
            i = j + 1
        elif c == "{":
            j = _matching(s, i, "{", "}")
            out["conditionAction"] = s[i + 1 : j].strip()
            i = j + 1
        elif c == "/":
            out["action"] = s[i + 1 :].strip()
            break
        else:
            raise BlockError(f"the transition label '{label}' cannot be read near '{s[i:i + 10]}'")
    return out


# -------------------------------------------------------------------------------------
# The chart description
# -------------------------------------------------------------------------------------


class SState:
    def __init__(self, name: str, parent: "SState | None", decomposition: str = "exclusive"):
        self.name = name
        self.parent = parent
        self.decomposition = decomposition
        self.children: list[SState] = []
        self.entry = self.during = self.exit = ""
        self.outgoing: list[STrans] = []
        self.defaults: list[STrans] = []  # default transitions into its children
        self.active = False
        self.active_child: SState | None = None
        self.entered_tick = 0
        self.entered_time = 0.0
        self.last_every: dict = {}

    @property
    def path(self) -> str:
        parts = []
        s: SState | None = self
        while s is not None and s.parent is not None:
            parts.append(s.name)
            s = s.parent
        return ".".join(reversed(parts))

    def ancestors(self) -> list["SState"]:
        out = []
        s = self.parent
        while s is not None:
            out.append(s)
            s = s.parent
        return out


class SJunction:
    def __init__(self, name: str, parent: SState):
        self.name = name
        self.parent = parent
        self.outgoing: list[STrans] = []
        self.path = name


class STrans:
    def __init__(self, src, dst, label: str, order: int, parent: SState):
        self.src, self.dst, self.label, self.order, self.parent = src, dst, label, order, parent
        parts = parse_transition_label(label)
        self.events: list[str] = []
        self.temporal_events: list[str] = []
        for e in parts["events"]:
            (self.temporal_events if TEMPORAL.match(e) else self.events).append(e)
        cond = parts["condition"]
        if self.temporal_events:
            cond = " && ".join([f"({e})" for e in self.temporal_events] + ([f"({cond})"] if cond else []))
        self.condition = cond
        self.condition_action = parts["conditionAction"]
        self.action = parts["action"]


class ChartDef:
    """The parsed chart: states, junctions, transitions, data and events."""

    def __init__(self):
        self.root = SState("", None)
        self.states: dict[str, SState] = {}
        self.junctions: dict[str, SJunction] = {}
        self.data: list[dict] = []
        self.events: list[dict] = []

    @classmethod
    def from_json(cls, d: Any) -> "ChartDef":
        """Read the ``"chart"`` description of a Chart block.

        ``{"decomposition": "exclusive"|"parallel",
           "data": [{"name", "scope": "input"|"output"|"local"|"constant"|"parameter", "initial"}],
           "events": [{"name", "scope": "input"|"output"|"local", "trigger": "rising"|"falling"|"either"|
                       "function-call"}],
           "states": [{"name", "label", "decomposition", "states": [...]}],
           "junctions": [{"name", "parent"}],
           "transitions": [{"from", "to", "label"}]}``

        States are named by their path with dots ("On.Fast"); "from": "" is a default transition.
        """
        if not isinstance(d, dict):
            raise BlockError("the chart description is missing (the block has no \"chart\" entry)")
        c = cls()
        c.root.decomposition = _decomp(d.get("decomposition", "exclusive"))

        def add_states(items, parent):
            for s in items or []:
                if not isinstance(s, dict):
                    raise BlockError("every state must be an object with a name")
                label = str(s.get("label", "") or "")
                parsed = parse_state_label(label) if label else {"name": "", "entry": "", "during": "", "exit": ""}
                name = str(s.get("name") or parsed["name"] or "").strip()
                if not re.fullmatch(r"[A-Za-z_]\w*", name):
                    raise BlockError(f"'{name}' is not a valid state name (letters, digits and _, not first a digit)")
                st = SState(name, parent, _decomp(s.get("decomposition", "exclusive")))
                st.entry = "\n".join(x for x in (parsed["entry"], s.get("entry", "")) if x)
                st.during = "\n".join(x for x in (parsed["during"], s.get("during", "")) if x)
                st.exit = "\n".join(x for x in (parsed["exit"], s.get("exit", "")) if x)
                if any(x.name == name for x in parent.children):
                    raise BlockError(f"two states are named '{st.path}'")
                parent.children.append(st)
                c.states[st.path] = st
                add_states(s.get("states"), st)

        add_states(d.get("states"), c.root)
        for j in d.get("junctions") or []:
            name = str(j.get("name", "")).strip()
            if not name or name in c.junctions or name in c.states:
                raise BlockError(f"junction names must be unique and differ from state names ('{name}')")
            parent = c.states.get(str(j.get("parent", "") or ""), c.root)
            c.junctions[name] = SJunction(name, parent)
        for k, t in enumerate(d.get("transitions") or []):
            src = c.find(str(t.get("from", "") or ""), allow_none=True)
            dst = c.find(str(t.get("to", "") or ""))
            label = str(t.get("label", "") or "")
            order = int(t.get("order", k + 1))
            if src is None:
                parent = dst.parent if dst.parent is not None else c.root
                if "parent" in t:
                    parent = c.states.get(str(t["parent"]), c.root)
                tr = STrans(None, dst, label, order, parent)
                parent.defaults.append(tr)
            else:
                tr = STrans(src, dst, label, order, src.parent)
                src.outgoing.append(tr)
        for obj in list(c.states.values()) + list(c.junctions.values()) + [c.root]:
            obj.outgoing.sort(key=lambda tr: tr.order)
            if isinstance(obj, SState):
                obj.defaults.sort(key=lambda tr: tr.order)
        for s in [c.root] + list(c.states.values()):
            if s.children and s.decomposition == "exclusive" and not s.defaults:
                where = f"state '{s.path}'" if s.parent else "the chart"
                raise BlockError(f"{where} has exclusive substates but no default transition into one of them")
        names = set()
        for item in d.get("data") or []:
            name = str(item.get("name", "")).strip()
            scope = str(item.get("scope", "local")).strip().lower()
            if not re.fullmatch(r"[A-Za-z_]\w*", name) or name in names:
                raise BlockError(f"data names must be unique variable names ('{name}')")
            if scope not in ("input", "output", "local", "constant", "parameter"):
                raise BlockError(f"data '{name}' has an unknown scope '{scope}'")
            names.add(name)
            c.data.append({"name": name, "scope": scope, "initial": str(item.get("initial", "") or "")})
        for item in d.get("events") or []:
            name = str(item.get("name", "")).strip()
            scope = str(item.get("scope", "local")).strip().lower()
            trig = str(item.get("trigger", "either")).strip().lower().replace("_", "-")
            if not re.fullmatch(r"[A-Za-z_]\w*", name) or name in names:
                raise BlockError(f"event names must be unique and differ from data names ('{name}')")
            if scope not in ("input", "output", "local"):
                raise BlockError(f"event '{name}' has an unknown scope '{scope}'")
            if trig in ("functioncall", "function call"):
                trig = "function-call"
            if trig not in ("rising", "falling", "either", "function-call"):
                raise BlockError(f"event '{name}' has an unknown trigger '{trig}'")
            names.add(name)
            c.events.append({"name": name, "scope": scope, "trigger": trig})
        return c

    def find(self, name: str, allow_none: bool = False):
        name = name.strip()
        if not name:
            if allow_none:
                return None
            raise BlockError("a transition needs a destination")
        if name in self.states:
            return self.states[name]
        if name in self.junctions:
            return self.junctions[name]
        raise BlockError(f"a transition refers to '{name}', which is not a state or junction of the chart")

    def ports(self) -> tuple[list[str], list[str]]:
        """Names of the input and output ports."""
        ins = [d["name"] for d in self.data if d["scope"] == "input"]
        if any(e["scope"] == "input" for e in self.events):
            ins.append("events")
        outs = [d["name"] for d in self.data if d["scope"] == "output"]
        outs += [e["name"] for e in self.events if e["scope"] == "output"]
        return ins, outs


def _decomp(v: Any) -> str:
    v = str(v or "exclusive").strip().lower()
    if v in ("parallel", "and", "set"):
        return "parallel"
    if v in ("exclusive", "or", "cluster"):
        return "exclusive"
    raise BlockError(f"unknown decomposition '{v}' (exclusive or parallel)")


def chart_ports(chart: Any) -> tuple[int, int]:
    """Numbers of input and output ports of a chart description (0, 0 if it cannot be read)."""
    if not isinstance(chart, dict):
        return 0, 0
    ins = sum(1 for d in chart.get("data") or [] if str(d.get("scope", "")).lower() == "input")
    ins += 1 if any(str(e.get("scope", "")).lower() == "input" for e in chart.get("events") or []) else 0
    outs = sum(1 for d in chart.get("data") or [] if str(d.get("scope", "")).lower() == "output")
    outs += sum(1 for e in chart.get("events") or [] if str(e.get("scope", "")).lower() == "output")
    return ins, outs


# -------------------------------------------------------------------------------------
# Running a chart
# -------------------------------------------------------------------------------------


class _Taken(Exception):
    pass


class ChartRun:
    """The state of one chart while the model runs, and the code that steps it."""

    def __init__(self, cdef: ChartDef, ctx: Any, path: str):
        from ..interpreter import Frame

        self.c = cdef
        self.ctx = ctx
        self.path = path
        interp = ctx.interp
        self.interp = interp
        src = interp.base
        self.frame = Frame(interp, src.functions, name=f"chart {path}", file=src.file)
        self.frame.vars = dict(src.vars)
        self.frame.vars.update(ctx.scope)
        self.programs: dict[str, Any] = {}
        self.exprs: dict[str, Any] = {}
        self.ticks = 0
        self.t = 0.0
        self.t_prev = None
        self.started = False
        self.out_events: dict[str, int] = {}
        self.depth = 0
        for e in cdef.events:
            self.frame.vars[e["name"]] = _char(e["name"])
        for d in cdef.data:
            if d["scope"] == "parameter":
                self.frame.vars[d["name"]] = ctx.evaluate(d["name"])
            elif d["scope"] != "input":
                expr = d["initial"] or "0"
                try:
                    self.frame.vars[d["name"]] = ctx.evaluate(expr)
                except Exception as exc:
                    raise BlockError(
                        f"the initial value '{expr}' of '{d['name']}' cannot be evaluated: "
                        f"{getattr(exc, 'message', exc)}"
                    ) from None
            else:
                self.frame.vars[d["name"]] = np.zeros((1, 1))
        for v in ("tick", "sec", "msec", "usec"):
            self.frame.vars.setdefault(v, _char(v))

    # -- code ----------------------------------------------------------------------------
    def _prepare(self, text: str, expr: bool) -> tuple[Any, list]:
        """Parse an action or condition once. Temporal operators and in(...) become variables set before each
        evaluation."""
        from ..parser import parse

        specials = []
        out = []
        i = 0
        names = {e["name"] for e in self.c.events}
        while True:
            m = TEMPORAL.search(text, i)
            m2 = IN_STATE.search(text, i)
            if m2 and (not m or m2.start() < m.start()):
                var = f"mylab_in{len(specials)}"
                specials.append(("in", m2.group(1)))
                out.append(text[i : m2.start()] + var)
                i = m2.end()
                continue
            if not m:
                break
            j = _matching(text, m.end() - 1, "(", ")")
            inner = text[m.end() : j]
            args = _split_args(inner)
            if len(args) != 2:
                raise BlockError(f"'{text[m.start():j + 1]}' needs two arguments: a number and tick or sec")
            unit = args[1].strip()
            if unit not in ("tick", "sec", "msec", "usec") and unit not in names:
                raise BlockError(f"'{text[m.start():j + 1]}': the second argument must be tick, sec, msec or usec")
            var = f"mylab_tl{len(specials)}"
            specials.append((m.group(1), args[0].strip(), unit))
            out.append(text[i : m.start()] + var)
            i = j + 1
        out.append(text[i:])
        code = "".join(out)
        if not expr and names:  # a statement that is only an event name broadcasts it
            code = "\n".join(
                re.sub(rf"^(\s*)({'|'.join(map(re.escape, names))})\s*;?\s*$", r"\1send(\2);", line)
                for line in code.split("\n")
            )
        try:
            if expr:
                from .engine import Context

                node = Context._parse_expr(code)
            else:
                node = parse(code)
        except Exception as exc:
            what = "condition" if expr else "action"
            raise BlockError(f"the {what} '{text}' cannot be read: {getattr(exc, 'message', exc)}") from None
        return node, specials

    def _set_specials(self, specials: list, owner: SState) -> None:
        for k, sp in enumerate(specials):
            if sp[0] == "in":
                st = self.c.states.get(sp[1])
                if st is None:
                    raise BlockError(f"in({sp[1]}): there is no state '{sp[1]}'")
                self.frame.vars[f"mylab_in{k}"] = np.array([[st.active]], dtype=bool)
                continue
            op, n_expr, unit = sp
            n = float(np.ravel(sig(self._eval_text(n_expr)))[0])
            if unit == "tick" or unit in {e["name"] for e in self.c.events}:
                count = float(self.ticks - owner.entered_tick)
                prev = count - 1
            else:
                scale = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}[unit]
                n = n * scale
                count = self.t - owner.entered_time
                prev = (self.t_prev - owner.entered_time) if self.t_prev is not None else -1.0
            eps = 1e-9 * max(1.0, abs(n), abs(self.t))
            if op == "after":
                hit = count >= n - eps
            elif op == "before":
                hit = count < n - eps
            elif op == "at":
                hit = abs(count - n) <= eps if unit == "tick" else (prev < n - eps <= count)
            else:  # every
                if n <= 0:
                    raise BlockError("every(n, ...) needs a positive n")
                if unit == "tick":
                    hit = count > 0 and abs(count / n - round(count / n)) < 1e-9
                else:
                    hit = count > eps and np.floor(count / n + 1e-9) > np.floor(max(prev, 0) / n + 1e-9)
                    if prev < 0:
                        hit = False
            self.frame.vars[f"mylab_tl{k}"] = np.array([[bool(hit)]])

    def _eval_text(self, text: str) -> Any:
        from .engine import Context

        node = Context._parse_expr(text)
        return self._eval_node(node)

    def _eval_node(self, node: Any) -> Any:
        interp = self.interp
        interp.frames.append(self.frame)
        try:
            return interp.eval(node, self.frame)
        finally:
            interp.frames.pop()

    def run(self, code: str, owner: SState, what: str) -> None:
        if not code or not code.strip():
            return
        from ..errors import MatlabError

        if code not in self.programs:
            self.programs[code] = self._prepare(code, False)
        program, specials = self.programs[code]
        self._set_specials(specials, owner)
        interp = self.interp
        interp.frames.append(self.frame)
        out = interp.out
        interp.out = lambda s: None  # a missing semicolon in an action does not print at every step
        try:
            interp.run_program(program, self.frame)
        except MatlabError as exc:
            raise BlockError(f"{what}: {exc.message}") from None
        finally:
            interp.frames.pop()
            interp.out = out

    def test(self, tr: STrans, owner: SState) -> bool:
        if not tr.condition:
            return True
        from ..errors import MatlabError
        from ..values import is_true

        if tr.condition not in self.exprs:
            self.exprs[tr.condition] = self._prepare(tr.condition, True)
        node, specials = self.exprs[tr.condition]
        self._set_specials(specials, owner)
        try:
            return bool(is_true(self._eval_node(node)))
        except MatlabError as exc:
            raise BlockError(f"the condition [{tr.condition}] of {_describe(tr)}: {exc.message}") from None

    # -- events -------------------------------------------------------------------------
    def send(self, name: str) -> None:
        ev = next((e for e in self.c.events if e["name"] == name), None)
        if ev is None:
            raise BlockError(f"send({name}): the chart has no event '{name}'")
        if ev["scope"] == "output":
            self.out_events[name] = self.out_events.get(name, 0) + 1
        elif ev["scope"] == "local":
            if self.depth > 20:
                raise BlockError(f"the event '{name}' is sent recursively more than 20 levels deep")
            self.depth += 1
            try:
                self.process(self.c.root, name)
            finally:
                self.depth -= 1
        else:
            raise BlockError(f"send({name}): '{name}' is an input event; it cannot be sent from inside the chart")

    # -- structure ----------------------------------------------------------------------
    def enter(self, s: SState, ev: Any) -> None:
        p = s.parent
        s.active = True
        s.entered_tick = self.ticks
        s.entered_time = self.t
        if p is not None and p.decomposition == "exclusive":
            p.active_child = s
        self.run(s.entry, s, f"the entry action of state '{s.path}'")
        if s.active:
            self.enter_children(s, ev)

    def enter_children(self, s: SState, ev: Any) -> None:
        if s.decomposition == "parallel":
            for ch in s.children:
                if not ch.active:
                    self.enter(ch, ev)
            return
        if not s.defaults:
            return
        path = self.find_path(s.defaults, ev, s)
        if path is None:
            if s.children:
                raise BlockError(
                    f"no default transition of {'state ' + repr(s.path) if s.parent else 'the chart'} could be taken"
                )
            return
        for tr in path:
            self.run(tr.action, s, f"the transition action of {_describe(tr)}")
        end = path[-1].dst
        if isinstance(end, SState):
            self.enter_down(s, end, ev)

    def enter_down(self, anc: SState, dst: SState, ev: Any) -> None:
        chain = []
        x: SState | None = dst
        while x is not None and x is not anc:
            chain.append(x)
            x = x.parent
        chain.reverse()
        for node in chain[:-1]:
            if not node.active:
                node.active = True
                node.entered_tick, node.entered_time = self.ticks, self.t
                if node.parent is not None and node.parent.decomposition == "exclusive":
                    node.parent.active_child = node
                self.run(node.entry, node, f"the entry action of state '{node.path}'")
        if chain:
            self.enter(chain[-1], ev)
            # parallel siblings along the way start too
            for node in chain:
                p = node.parent
                if p is not None and p.decomposition == "parallel" and p.active:
                    for sib in p.children:
                        if not sib.active:
                            self.enter(sib, ev)

    def exit(self, s: SState) -> None:
        if not s.active:
            return
        if s.decomposition == "parallel":
            for ch in reversed(s.children):
                self.exit(ch)
        elif s.active_child is not None:
            self.exit(s.active_child)
        self.run(s.exit, s, f"the exit action of state '{s.path}'")
        s.active = False
        p = s.parent
        if p is not None and p.active_child is s:
            p.active_child = None

    # -- transitions --------------------------------------------------------------------
    def valid(self, tr: STrans, ev: Any, owner: SState) -> bool:
        if tr.events and ev not in tr.events:
            return False
        return self.test(tr, owner)

    def find_path(self, transitions: list, ev: Any, owner: SState, depth: int = 0) -> list | None:
        if depth > 200:
            raise BlockError("a path of transitions through junctions is longer than 200 steps (a loop?)")
        for tr in transitions:
            if not self.valid(tr, ev, owner):
                continue
            self.run(tr.condition_action, owner, f"the condition action of {_describe(tr)}")
            if isinstance(tr.dst, SState):
                return [tr]
            if not tr.dst.outgoing:  # a junction with no way on ends the flow here
                return [tr]
            rest = self.find_path(tr.dst.outgoing, ev, owner, depth + 1)
            if rest is not None:
                return [tr] + rest
        return None

    def take(self, src: SState, path: list, ev: Any) -> bool:
        """Carry out a path of transitions that starts at an active state. True if the state was left."""
        end = path[-1].dst
        if not isinstance(end, SState):
            for tr in path:
                self.run(tr.action, src, f"the transition action of {_describe(tr)}")
            return False
        lca = _common(src, end)
        top = src
        while top.parent is not lca:
            top = top.parent
        self.exit(top)
        for tr in path:
            self.run(tr.action, src, f"the transition action of {_describe(tr)}")
        if lca.active or lca.parent is None:
            self.enter_down(lca, end, ev)
        return True

    def step_state(self, s: SState, ev: Any) -> None:
        if s.outgoing:
            path = self.find_path(s.outgoing, ev, s)
            if path is not None and self.take(s, path, ev):
                return
        if not s.active:
            return
        self.run(s.during, s, f"the during action of state '{s.path}'")
        if s.active:
            self.process(s, ev)

    def process(self, s: SState, ev: Any) -> None:
        if s.decomposition == "parallel":
            for ch in list(s.children):
                if not s.active and s.parent is not None:
                    return
                if ch.active:
                    self.step_state(ch, ev)
        elif s.active_child is not None:
            self.step_state(s.active_child, ev)
        elif not s.children and s.defaults and s.parent is None:
            self.enter_children(s, ev)  # a flow chart: the default paths run at every step

    # -- one step ----------------------------------------------------------------------
    def step(self, t: float, ev: Any) -> None:
        self.t = t
        self.ticks += 1
        if not self.started:
            self.started = True
            self.c.root.active = True
            self.enter_children(self.c.root, ev)
        else:
            self.process(self.c.root, ev)
        self.t_prev = t


def _common(a: SState, b: SState) -> SState:
    """The innermost state that contains both (strictly: a state is not its own container)."""
    up_a = a.ancestors()
    up_b = set(map(id, b.ancestors()))
    for s in up_a:
        if id(s) in up_b:
            return s
    return up_a[-1]


def _describe(tr: STrans) -> str:
    src = "(default)" if tr.src is None else tr.src.path
    return f"the transition {src} -> {tr.dst.path}" + (f" '{tr.label}'" if tr.label else "")


def _split_args(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _char(s: str) -> Any:
    from ..values import char

    return char(s)


# -------------------------------------------------------------------------------------
# The block
# -------------------------------------------------------------------------------------


@block(
    "Chart",
    "User-defined",
    "A state chart: states with entry, during and exit actions, transitions with events and conditions.",
    title="Chart",
    icon="chart",
)
class Chart(Node):
    params = [P("SampleTime", "-1", "Sample time (s); -1 takes it from the inputs")]
    nin = "chart"
    nout = "chart"
    time_varying = True

    def __init__(self, path, raw, ctx):
        super().__init__(path, raw, ctx)
        self.cdef_json = None

    def _json(self):
        if self.cdef_json is not None:
            return self.cdef_json
        blk = getattr(self, "block", None)
        return blk.extra.get("chart") if blk is not None else None

    def n_in(self):
        return chart_ports(self._json())[0]

    def n_out(self):
        return chart_ports(self._json())[1]

    def init(self):
        blk = getattr(self, "block", None)
        self.cdef_json = blk.extra.get("chart") if blk is not None else self.cdef_json
        self.cdef = ChartDef.from_json(self.cdef_json)
        self.run_state = ChartRun(self.cdef, self.ctx, self.path)
        c = self.cdef
        self.in_data = [d["name"] for d in c.data if d["scope"] == "input"]
        self.out_data = [d["name"] for d in c.data if d["scope"] == "output"]
        self.in_events = [e for e in c.events if e["scope"] == "input"]
        self.out_events = [e for e in c.events if e["scope"] == "output"]
        self.prev_events = None
        self.edge_state = {e["name"]: 0.0 for e in self.out_events}
        self.held = None

    def sample_time(self):
        if self.in_events:
            return None
        return self.read_sample("SampleTime")

    def _outputs(self) -> list:
        vals = []
        for name in self.out_data:
            try:
                vals.append(sig(self.run_state.frame.vars[name]))
            except BlockError as exc:
                raise BlockError(f"the output '{name}': {exc}") from None
        for e in self.out_events:
            n = self.run_state.out_events.get(e["name"], 0)
            if e["trigger"] == "function-call":
                vals.append(np.array([float(n)]))
            else:
                if n % 2:
                    self.edge_state[e["name"]] = 1.0 - self.edge_state[e["name"]]
                vals.append(np.array([self.edge_state[e["name"]]]))
        return vals

    def out(self, t, u):
        sim = self.sim
        if sim.minor or sim.initializing:
            if self.held is None:
                self.held = self._outputs()
            return list(self.held)
        rs = self.run_state
        frame = rs.frame
        for name, v in zip(self.in_data, u):
            frame.vars[name] = to_value(np.asarray(v, dtype=float))
        rs.out_events = {}
        state = self.ctx.interp.state
        previous = state.get("chart_send")
        state["chart_send"] = rs.send
        try:
            if self.in_events:
                v = np.ravel(u[len(self.in_data)]) if len(u) > len(self.in_data) else np.zeros(0)
                if v.size not in (1, len(self.in_events)):
                    raise BlockError(
                        f"the event input has {v.size} elements but the chart has {len(self.in_events)} input events"
                    )
                v = np.broadcast_to(v, (len(self.in_events),)) if v.size else np.zeros(len(self.in_events))
                prev = self.prev_events
                for k, e in enumerate(self.in_events):
                    if e["trigger"] == "function-call":
                        link = self.inputs[len(self.in_data)] if len(self.inputs) > len(self.in_data) else None
                        fresh = link is not None and link[0].stamp == sim.step_no
                        times = int(round(float(v[k]))) if fresh else 0
                    else:
                        p = None if prev is None else prev[k]
                        up = p is not None and p <= 0 < v[k]
                        down = p is not None and p > 0 >= v[k]
                        hit = up if e["trigger"] == "rising" else down if e["trigger"] == "falling" else up or down
                        times = 1 if hit else 0
                    for _ in range(max(times, 0)):
                        rs.step(t, e["name"])
                self.prev_events = np.array(v, dtype=float)
            else:
                rs.step(t, None)
        finally:
            if previous is None:
                state.pop("chart_send", None)
            else:
                state["chart_send"] = previous
        self.held = self._outputs()
        return list(self.held)

    def active_states(self) -> list[str]:
        return [p for p, s in self.cdef.states.items() if s.active]


# -------------------------------------------------------------------------------------
# Building charts from code
# -------------------------------------------------------------------------------------


class ChartBuilder:
    """Makes the ``"chart"`` description of a Chart block.

    >>> c = ChartBuilder()
    >>> c.data("u", "input"); c.data("y", "output", "0")
    >>> c.state("Off", "entry: y = 0;"); c.state("On", "entry: y = 1;")
    >>> c.transition("", "Off"); c.transition("Off", "On", "[u > 1]"); c.transition("On", "Off", "[u < 0]")
    >>> block.extra["chart"] = c.to_json()
    """

    def __init__(self, decomposition: str = "exclusive"):
        self.d: dict = {"decomposition": _decomp(decomposition), "data": [], "events": [], "states": [],
                        "junctions": [], "transitions": []}

    def state(self, path: str, label: str = "", decomposition: str = "exclusive") -> "ChartBuilder":
        parts = path.split(".")
        items = self.d["states"]
        for p in parts[:-1]:
            parent = next((s for s in items if s["name"] == p), None)
            if parent is None:
                raise BlockError(f"there is no state '{p}' to put '{path}' into")
            items = parent.setdefault("states", [])
        entry = {"name": parts[-1]}
        if label:
            entry["label"] = label
        if _decomp(decomposition) == "parallel":
            entry["decomposition"] = "parallel"
        items.append(entry)
        return self

    def junction(self, name: str, parent: str = "") -> "ChartBuilder":
        j = {"name": name}
        if parent:
            j["parent"] = parent
        self.d["junctions"].append(j)
        return self

    def transition(self, src: str, dst: str, label: str = "") -> "ChartBuilder":
        t = {"from": src, "to": dst}
        if label:
            t["label"] = label
        self.d["transitions"].append(t)
        return self

    def data(self, name: str, scope: str = "local", initial: str = "") -> "ChartBuilder":
        item = {"name": name, "scope": scope}
        if initial != "":
            item["initial"] = str(initial)
        self.d["data"].append(item)
        return self

    def event(self, name: str, scope: str = "local", trigger: str = "either") -> "ChartBuilder":
        self.d["events"].append({"name": name, "scope": scope, "trigger": trigger})
        return self

    def to_json(self) -> dict:
        out = {k: v for k, v in self.d.items() if v not in ([], "")}
        out["decomposition"] = self.d["decomposition"]
        ChartDef.from_json(out)  # check it
        return out


# -------------------------------------------------------------------------------------
# Charts in model files (.slx and .mdl Stateflow parts)
# -------------------------------------------------------------------------------------
#
# A Stateflow part lists objects linked by id: a chart, its states (labelString, treeNode
# [parent firstChild previous next], decomposition), transitions (labelString, src and dst ids,
# linkNode [parent ...], executionOrder), junctions (linkNode) and data (name, scope). This
# layout was seen in public files (where the charts hold MATLAB Function code); files with
# state charts were not available, so the names of scopes and decompositions other than the
# ones seen (CLUSTER_* exclusive, INPUT_DATA, OUTPUT_DATA) follow the same pattern but are
# not confirmed.

_SCOPES = {"INPUT": "input", "OUTPUT": "output", "LOCAL": "local", "CONSTANT": "constant", "PARAMETER": "parameter"}


def _ptext(el, name: str) -> str:
    for p in el.findall("P"):
        if p.get("Name") == name:
            return p.text or ""
    return ""


def _ids(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,\[\]]+", text or "") if t]


def chart_from_sf(chart_el) -> dict:
    """The chart description of a Chart block from the Stateflow objects of its chart."""
    objs = {}
    for kind in ("state", "transition", "junction", "data", "event"):
        for el in chart_el.iter(kind):
            objs.setdefault(kind, []).append(el)
    cid = chart_el.get("id") or _ptext(chart_el, "id")

    def decomp(text: str) -> str:
        return "parallel" if "SET" in (text or "").upper() or "AND" in (text or "").upper() else "exclusive"

    states = {}
    for el in objs.get("state", []):
        sid = el.get("id") or _ptext(el, "id")
        if "FUNC" in _ptext(el, "type").upper():
            continue
        tree = _ids(_ptext(el, "treeNode"))
        parent = tree[0] if tree else cid
        label = _ptext(el, "labelString")
        name = parse_state_label(label)["name"] or f"S{sid}"
        states[sid] = {"id": sid, "parent": parent, "label": label, "name": name,
                       "decomposition": decomp(_ptext(el, "decomposition"))}

    def path(sid: str) -> str:
        parts = []
        while sid in states:
            parts.append(states[sid]["name"])
            sid = states[sid]["parent"]
        return ".".join(reversed(parts))

    def nest(parent: str) -> list:
        out = []
        for sid, s in states.items():
            if s["parent"] == parent:
                lines = s["label"].split("\n", 1)
                entry = {"name": s["name"]}
                if len(lines) > 1 and lines[1].strip():
                    entry["label"] = lines[1]
                if s["decomposition"] == "parallel":
                    entry["decomposition"] = "parallel"
                kids = nest(sid)
                if kids:
                    entry["states"] = kids
                out.append(entry)
        return out

    junctions = {}
    for el in objs.get("junction", []):
        jid = el.get("id") or _ptext(el, "id")
        link = _ids(_ptext(el, "linkNode"))
        name = _ptext(el, "name") or f"j{jid}"  # files mylab writes keep the junction's name
        junctions[jid] = {"name": name, "parent": path(link[0]) if link and link[0] in states else ""}

    def where(el) -> str:
        sid = el.get("id") if el is not None else None
        if sid is None and el is not None:
            sid = _ptext(el, "id")
        if not sid:
            return ""
        if sid in states:
            return path(sid)
        if sid in junctions:
            return junctions[sid]["name"]
        return ""

    transitions = []
    for el in objs.get("transition", []):
        src, dst = el.find("src"), el.find("dst")
        t = {"from": where(src), "to": where(dst), "label": _ptext(el, "labelString")}
        order = _ptext(el, "executionOrder")
        if order.strip().isdigit():
            t["order"] = int(order)
        link = _ids(_ptext(el, "linkNode"))
        if not t["from"] and link and link[0] in states:
            t["parent"] = path(link[0])
        if not t["label"]:
            del t["label"]
        if t["to"]:
            transitions.append(t)
    # transitions leaving the same place are tried by their execution order; the list order says that
    first: dict[str, int] = {}
    for n, t in enumerate(transitions):
        first.setdefault(t["from"], n)
    transitions.sort(key=lambda t: (first[t["from"]], t.get("order", 0)))
    for t in transitions:
        t.pop("order", None)
    data = []
    for el in objs.get("data", []):
        scope = _ptext(el, "scope").upper()
        kind = next((v for k, v in _SCOPES.items() if k in scope), "local")
        item = {"name": _ptext(el, "name"), "scope": kind}
        init = _ptext(el, "initialValue") or next((p.text for p in el.iter("P") if p.get("Name") == "initialValue"), "")
        if init:
            item["initial"] = init
        data.append(item)
    events = []
    for el in objs.get("event", []):
        scope = _ptext(el, "scope").upper()
        kind = "input" if "INPUT" in scope else "output" if "OUTPUT" in scope else "local"
        trig = _ptext(el, "trigger").upper()
        trigger = ("function-call" if "FUNCTION" in trig else "rising" if "RISING" in trig
                   else "falling" if "FALLING" in trig else "either")
        events.append({"name": _ptext(el, "name"), "scope": kind, "trigger": trigger})
    out = {
        "decomposition": decomp(_ptext(chart_el, "decomposition")),
        "data": data,
        "events": events,
        "states": nest(cid),
        "junctions": [{k: v for k, v in j.items() if v} for j in junctions.values()],
        "transitions": transitions,
    }
    return {k: v for k, v in out.items() if v != []}


def chart_to_sf(chart: dict, path: str, first_id: int) -> tuple[list, int]:
    """The Stateflow objects of a chart description, as (kind, id, values, children) tuples; and the next free id.

    The inverse of :func:`chart_from_sf`: states with labelString and treeNode, transitions with src and dst,
    junctions, data and events.
    """
    k = first_id
    cid = k
    objects: list = []
    ids: dict[str, int] = {}
    decomp = "SET_CHART" if chart.get("decomposition") == "parallel" else "CLUSTER_CHART"
    head = [("name", path), ("decomposition", decomp)]

    def add_states(items, parent_id, prefix):
        nonlocal k
        for s in items or []:
            k += 1
            sid = k
            p = f"{prefix}{s['name']}"
            ids[p] = sid
            label = s["name"] + ("\n" + s["label"] if s.get("label") else "")
            for key in ("entry", "during", "exit"):
                if s.get(key):
                    label += f"\n{key}: {s[key]}"
            dec = "SET_STATE" if s.get("decomposition") == "parallel" else "CLUSTER_STATE"
            objects.append(("state", sid, [("labelString", label), ("chart", str(cid)),
                                           ("treeNode", f"[{parent_id} 0 0 0]"), ("decomposition", dec),
                                           ("type", "OR_STATE")], []))
            add_states(s.get("states"), sid, p + ".")

    add_states(chart.get("states"), cid, "")
    for j in chart.get("junctions") or []:
        k += 1
        ids[j["name"]] = k
        parent = ids.get(j.get("parent", ""), cid)
        objects.append(("junction", k, [("name", j["name"]), ("chart", str(cid)), ("linkNode", f"[{parent} 0 0]"),
                                        ("type", "CONNECTIVE_JUNCTION")], []))
    for n, t in enumerate(chart.get("transitions") or []):
        k += 1
        src = ids.get(t.get("from", ""))
        dst = ids.get(t.get("to", ""))
        parent = ids.get(t.get("parent", ""), cid)
        if src is None and dst is not None:
            to = t.get("to", "")
            parent = ids.get(to.rsplit(".", 1)[0], cid) if "." in to and "parent" not in t else parent
        kids = [("src", [("id", str(src))] if src is not None else []), ("dst", [("id", str(dst))])]
        objects.append(("transition", k, [("labelString", t.get("label", "")), ("chart", str(cid)),
                                          ("linkNode", f"[{parent} 0 0]"),
                                          ("executionOrder", str(t.get("order", n + 1)))], kids))
    for d in chart.get("data") or []:
        k += 1
        vals = [("name", d["name"]), ("scope", d.get("scope", "local").upper() + "_DATA"), ("linkNode", f"[{cid} 0 0]")]
        if d.get("initial"):
            vals.append(("initialValue", d["initial"]))
        objects.append(("data", k, vals, []))
    for e in chart.get("events") or []:
        k += 1
        trig = {"function-call": "FUNCTION_CALL_EVENT", "rising": "RISING_EDGE_EVENT",
                "falling": "FALLING_EDGE_EVENT"}.get(e.get("trigger", "either"), "EITHER_EDGE_EVENT")
        objects.append(("event", k, [("name", e["name"]), ("scope", e.get("scope", "local").upper() + "_EVENT"),
                                     ("trigger", trig), ("linkNode", f"[{cid} 0 0]")], []))
    return [("chart", cid, head, objects)], k + 1
