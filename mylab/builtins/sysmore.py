"""System functions: function handle details, memoize, valid names, threads, locks, the profiler,
futures, and clear errors for what a Python engine cannot do (Java, compiled MEX files).

Calling Python from MATLAB code (py.*, pyrun, pyrunfile, pyenv) is in ``pyinterop``.
"""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, class_name, is_char, string_result, to_int, to_str,
)
from ._registry import builtin
from ._util import bool_scalar, fscalar, is_text, nargs, num, text

# -- function handles ------------------------------------------------------------------------------


def _function_file(interp, name: str) -> str:
    found = interp.find_file_function(name)
    return found[1] if found is not None else ""


@builtin("functions")
def _functions(interp, args, nargout):
    nargs(args, 1, 1, "functions")
    f = args[0]
    if not isinstance(f, FunctionHandle):
        raise MatlabError("functions: the input must be a function handle.")
    from ..display import function_handle_text
    from ..interpreter import NestedRef
    if f.is_anonymous:
        captured = {k: v for k, v in f.closure.items() if k != "__functions__"}
        ws = Struct.scalar(captured) if captured else Struct()
        return Struct.scalar({"function": char(function_handle_text(f)), "type": char("anonymous"),
                              "file": char(""), "workspace": Cell.from_list([ws], column=True),
                              "within_file_path": char("")})
    name = str(f.name)
    local = f.closure.get("__functions__", {}).get(name)
    if isinstance(local, NestedRef):
        return Struct.scalar({"function": char(name), "type": char("nested"), "file": char(local.frame.file or ""),
                              "workspace": Cell.from_list([Struct.scalar(dict(local.frame.vars))], column=True)})
    if local is not None:
        return Struct.scalar({"function": char(name), "type": char("scopedfunction"),
                              "file": char(interp.frame.file or ""), "parentage": Cell.from_list([char(name)])})
    return Struct.scalar({"function": char(name), "type": char("simple"), "file": char(_function_file(interp, name))})


# -- memoize ---------------------------------------------------------------------------------------

def _freeze(v: Any) -> Any:
    """A hashable stand-in for a value, equal for equal inputs (NaNs compare equal)."""
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "O":
            return ("o", id(v))
        return ("a", v.dtype.str, v.shape, np.ascontiguousarray(v).tobytes())
    if isinstance(v, MString):
        return ("s", str(v))
    if isinstance(v, StringArray):
        return ("S", v.data.shape, tuple(v.data.flatten(order="F")))
    if isinstance(v, Cell):
        return ("c", v.data.shape, tuple(_freeze(x) for x in v.data.flatten(order="F")))
    if isinstance(v, Struct):
        return ("t", v.data.shape, tuple(v.fields),
                tuple(_freeze(d[f]) for d in v.data.flatten(order="F") for f in v.fields))
    if isinstance(v, FunctionHandle):
        return ("f", v.name) if not v.is_anonymous else ("f", id(v))
    if hasattr(v, "to_dense"):
        m = v.m
        return ("p", m.shape, m.dtype.str, m.indices.tobytes(), m.indptr.tobytes(), m.data.tobytes())
    return ("x", id(v))


class MemoizedFunction(MObject):
    """memoize(f): calls f once per distinct input list and remembers the outputs."""

    is_handle = True
    _METHODS = {"clearCache", "stats"}

    def __init__(self, fn: FunctionHandle):
        self.fn = fn
        self.enabled = True
        self.cache_size = 10
        self.cache: OrderedDict = OrderedDict()  # key -> [inputs, nargout, outputs, hits]
        self.hits = 0
        self.misses = 0

    def class_name(self) -> str:
        return "matlab.lang.MemoizedFunction"

    def isa_class(self, name: str) -> bool:
        return name in ("matlab.lang.MemoizedFunction", "handle")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def call_object(self, interp, args: list, nargout: int) -> list:
        if not self.enabled:
            return interp.call_function_value(self.fn, list(args), nargout)
        key = (tuple(_freeze(a) for a in args), max(nargout, 1))
        entry = self.cache.get(key)
        if entry is not None:
            entry[3] += 1
            self.hits += 1
            self.cache.move_to_end(key)
            return list(entry[2])
        outs = interp.call_function_value(self.fn, list(args), nargout)
        self.misses += 1
        self.cache[key] = [list(args), max(nargout, 1), list(outs), 0]
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # the entry used longest ago goes first
        return outs

    def index_get(self, interp, args: list) -> Any:
        outs = self.call_object(interp, args, 1)
        if not outs:
            raise MatlabError("The memoized function returned no value.")
        return outs[0]

    def get_field(self, interp, name: str) -> Any:
        if name == "Function":
            return self.fn
        if name == "Enabled":
            return bool_scalar(self.enabled)
        if name == "CacheSize":
            return fscalar(self.cache_size)
        raise MatlabError(f"Unrecognized property '{name}' for class 'matlab.lang.MemoizedFunction'.")

    def set_field(self, interp, name: str, value: Any) -> "MemoizedFunction":
        if name == "Enabled":
            self.enabled = bool(np.asarray(value).any())
        elif name == "CacheSize":
            n = to_int(value, "CacheSize")
            if n < 1:
                raise MatlabError("CacheSize must be a positive integer.")
            self.cache_size = n
            while len(self.cache) > n:
                self.cache.popitem(last=False)
        elif name == "Function":
            raise MatlabError("The Function property of a memoized function is read-only.")
        else:
            raise MatlabError(f"Unrecognized property '{name}' for class 'matlab.lang.MemoizedFunction'.")
        return self

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "clearCache":
            self.cache.clear()
            self.hits = self.misses = 0
            return []
        if name == "stats":
            return [self.stats()]
        return super().call_method(interp, name, args, nargout)

    def stats(self) -> Struct:
        entries = list(self.cache.values())
        cache = Struct.scalar({
            "Inputs": Cell.from_list([Cell.from_list(e[0]) for e in entries]),
            "Nargout": np.array([[float(e[1]) for e in entries]]).reshape(1, -1),
            "Outputs": Cell.from_list([Cell.from_list(e[2]) for e in entries]),
            "HitCount": np.array([[float(e[3]) for e in entries]]).reshape(1, -1),
            "TotalHits": fscalar(self.hits), "TotalMisses": fscalar(self.misses)})
        if entries:
            best = max(entries, key=lambda e: e[3])
            most = Struct.scalar({"Hits": fscalar(best[3]), "Input": Cell.from_list(best[0])})
        else:
            most = Struct.scalar({"Hits": fscalar(0), "Input": Cell.empty(0, 0)})
        calls = self.hits + self.misses
        return Struct.scalar({"Cache": cache, "MostHitCachedInput": most,
                              "CacheHitRatePercent": fscalar(100.0 * self.hits / calls if calls else 0.0),
                              "CacheOccupancyPercent": fscalar(100.0 * len(entries) / self.cache_size)})

    def display_lines(self) -> list[str]:
        from ..display import function_handle_text
        return ["  MemoizedFunction with properties:", "",
                f"     Function: {function_handle_text(self.fn)}",
                f"      Enabled: {int(self.enabled)}",
                f"    CacheSize: {self.cache_size}"]


@builtin("memoize")
def _memoize(interp, args, nargout):
    nargs(args, 1, 1, "memoize")
    f = args[0]
    if isinstance(f, MemoizedFunction):
        return f
    if not isinstance(f, FunctionHandle):
        raise MatlabError("memoize: the input must be a function handle.")
    table = interp.state.setdefault("memoized", {})
    key = ("name", f.name) if not f.is_anonymous else ("id", id(f))
    entry = table.get(key)
    if entry is None or (f.is_anonymous and entry.fn is not f):
        entry = table[key] = MemoizedFunction(f)  # the table also keeps the handle alive, so id() stays unique
    return entry


@builtin("clearAllMemoizedCaches")
def _clear_all_memoized(interp, args, nargout):
    for m in interp.state.get("memoized", {}).values():
        m.cache.clear()
        m.hits = m.misses = 0
    return None


# -- names -----------------------------------------------------------------------------------------

NAME_LENGTH_MAX = 63


@builtin("namelengthmax")
def _namelengthmax(interp, args, nargout):
    return float(NAME_LENGTH_MAX)


def _text_list(v: Any, what: str) -> tuple[list[str], str, tuple]:
    """(texts, kind, shape) of a char vector, string array or cell array of text."""
    if isinstance(v, MString):
        return [str(v)], "string", (1, 1)
    if isinstance(v, StringArray):
        if any(x is None for x in v.data.flat):
            raise MatlabError(f"{what} must not contain missing strings.")
        return list(v.data.flatten(order="F")), "string", v.data.shape
    if isinstance(v, Cell):
        return [to_str(x, what) for x in v.values_f()], "cell", v.data.shape
    if is_char(v) and v.shape[0] <= 1:
        return [to_str(v)], "char", (1, 1)
    raise MatlabError(f"{what} must be a character vector, a string array or a cell array of character vectors.")


def _text_result(items: list[str], kind: str, shape: tuple) -> Any:
    if kind == "char":
        return char(items[0])
    data = np.empty(len(items), dtype=object)
    for k, s in enumerate(items):
        data[k] = s if kind == "string" else char(s)
    data = data.reshape(shape, order="F")
    return string_result(data) if kind == "string" else Cell(data)


def _options(args: list, names: tuple, fn: str) -> dict:
    if len(args) % 2:
        raise MatlabError(f"{fn}: options must be name-value pairs.")
    out = {}
    for k in range(0, len(args), 2):
        key = text(args[k], "option name").lower()
        match = [n for n in names if n.lower() == key]
        if not match:
            raise MatlabError(f"{fn}: unknown option '{text(args[k])}'. Options are: {', '.join(names)}.")
        out[match[0]] = args[k + 1]
    return out


def make_valid_name(s: str, style: str, prefix: str) -> str:
    from ..lexer import KEYWORDS
    if re.fullmatch(r"[A-Za-z]\w*", s) and s not in KEYWORDS and len(s) <= NAME_LENGTH_MAX:
        return s
    if s in KEYWORDS:
        s = prefix + s[:1].upper() + s[1:]
    # a letter after white space becomes a capital and the white space goes
    s = re.sub(r"\s+([a-zA-Z])", lambda m: m.group(1).upper(), s.lstrip())
    s = re.sub(r"\s+", "", s)
    if style == "underscore":
        s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    elif style == "delete":
        s = re.sub(r"[^A-Za-z0-9_]", "", s)
    else:  # hex
        s = re.sub(r"[^A-Za-z0-9_]", lambda m: f"0x{ord(m.group(0)):02X}", s)
    if not s or not re.match(r"[A-Za-z]", s):
        s = prefix + s
    return s[:NAME_LENGTH_MAX]


@builtin("matlab.lang.makeValidName")
def _make_valid_name(interp, args, nargout):
    nargs(args, 1, None, "matlab.lang.makeValidName")
    items, kind, shape = _text_list(args[0], "The input")
    opts = _options(list(args[1:]), ("ReplacementStyle", "Prefix"), "matlab.lang.makeValidName")
    style = text(opts["ReplacementStyle"]).lower() if "ReplacementStyle" in opts else "underscore"
    if style not in ("underscore", "delete", "hex"):
        raise MatlabError("ReplacementStyle must be 'underscore', 'delete' or 'hex'.")
    prefix = text(opts["Prefix"]) if "Prefix" in opts else "x"
    if not re.fullmatch(r"[A-Za-z]\w*", prefix):
        raise MatlabError("Prefix must itself be a valid variable name.")
    out = [make_valid_name(s, style, prefix) for s in items]
    result = _text_result(out, kind, shape)
    if nargout > 1:
        changed = np.array([a != b for a, b in zip(out, items)], dtype=bool).reshape(shape, order="F")
        return result, changed
    return result


@builtin("matlab.lang.makeUniqueStrings")
def _make_unique_strings(interp, args, nargout):
    nargs(args, 1, 3, "matlab.lang.makeUniqueStrings")
    items, kind, shape = _text_list(args[0], "The input")
    exclude: list[str] = []
    check = list(range(len(items)))
    maxlen = NAME_LENGTH_MAX
    if len(args) > 1:
        a = args[1]  # names to avoid, or which elements may be changed (a logical or index vector)
        if isinstance(a, np.ndarray) and a.dtype == bool:
            flags = a.flatten(order="F")
            if flags.size != len(items):
                raise MatlabError("matlab.lang.makeUniqueStrings: the logical index must have one element per name.")
            check = [k for k in range(len(items)) if flags[k]]
        elif isinstance(a, np.ndarray) and a.dtype.kind in "iuf":
            check = [int(x) - 1 for x in num(a).flatten()] if a.size else check
            if any(k < 0 or k >= len(items) for k in check):
                raise MatlabError("matlab.lang.makeUniqueStrings: an index is out of range.")
        elif isinstance(a, Cell) and a.data.size == 0:
            exclude = []
        else:
            exclude = _text_list(a, "The names to avoid")[0]
    if len(args) > 2:
        maxlen = to_int(args[2], "maxStringLength")
        if maxlen < 1:
            raise MatlabError("matlab.lang.makeUniqueStrings: the maximum length must be positive.")
    out = list(items)
    fixed = set(exclude) | {out[k] for k in range(len(out)) if k not in check}
    seen: set[str] = set()
    changed = [False] * len(out)
    for k in check:
        s = out[k][:maxlen]
        if s != out[k]:
            changed[k] = True
        if s in fixed or s in seen:
            n = 1
            while True:
                tail = f"_{n}"
                cand = s[:maxlen - len(tail)] + tail
                if cand not in fixed and cand not in seen and cand not in items:
                    break
                n += 1
            s = cand
            changed[k] = True
        out[k] = s
        seen.add(s)
    result = _text_result(out, kind, shape)
    if nargout > 1:
        return result, np.array(changed, dtype=bool).reshape(shape, order="F")
    return result


# -- threads and locks -------------------------------------------------------------------------------

@builtin("maxNumCompThreads")
def _max_num_comp_threads(interp, args, nargout):
    nargs(args, 0, 1, "maxNumCompThreads")
    cores = float(os.cpu_count() or 1)
    current = float(interp.state.get("num_threads", cores))
    if args:
        if is_text(args[0]):
            if text(args[0]).lower() != "automatic":
                raise MatlabError("maxNumCompThreads: the argument must be a positive integer or 'automatic'.")
            interp.state.pop("num_threads", None)
        else:
            n = to_int(args[0], "the number of threads")
            if n < 1:
                raise MatlabError("maxNumCompThreads: the number of threads must be at least 1.")
            interp.state["num_threads"] = float(n)
    return current


def _function_in(interp, args: list, fn: str) -> str:
    if args:
        return text(args[0], "function name")
    frame = interp.frame
    if frame is interp.base:
        if fn == "mislocked":
            return ""
        raise MatlabError(f"{fn}: call {fn} inside a function, or name the function.")
    return frame.name


@builtin("mlock")
def _mlock(interp, args, nargout):
    nargs(args, 0, 0, "mlock")
    interp.state.setdefault("mlocked", set()).add(_function_in(interp, [], "mlock"))
    return None


@builtin("munlock")
def _munlock(interp, args, nargout):
    nargs(args, 0, 1, "munlock")
    interp.state.setdefault("mlocked", set()).discard(_function_in(interp, list(args), "munlock"))
    return None


@builtin("mislocked")
def _mislocked(interp, args, nargout):
    nargs(args, 0, 1, "mislocked")
    return bool_scalar(_function_in(interp, list(args), "mislocked") in interp.state.get("mlocked", set()))


# -- rats ------------------------------------------------------------------------------------------

@builtin("rats")
def _rats(interp, args, nargout):
    nargs(args, 1, 2, "rats")
    from .numlib import _rat_terms
    x = num(args[0])
    if x.dtype.kind == "c":
        raise MatlabError("rats: the input must be real.")
    x = x.astype(float)
    if x.ndim > 2:
        raise MatlabError("rats: the input must have at most two dimensions.")
    width = to_int(args[1], "the string length") if len(args) > 1 else 13
    if width < 3:
        raise MatlabError("rats: the string length must be at least 3.")
    finite = np.abs(x[np.isfinite(x)])
    tol = min(10.0 ** (-(width - 1) / 2) * float(finite.sum()), 0.1) if finite.size else 0.1
    left = width // 2 + 1  # room for the numerator; the denominator gets the rest
    right = width - left
    cells = []
    for v in x.flatten(order="C"):
        if np.isnan(v):
            t = "NaN".rjust(left) + " " * (right + 1)
        elif np.isinf(v):
            t = ("Inf" if v > 0 else "-Inf").rjust(left) + " " * (right + 1)
        else:
            n, d, _ = _rat_terms(float(v), tol)
            if d == 1 or n == 0:
                t = f"{n:.0f}".rjust(left) + " " * (right + 1)
            else:
                t = f"{n:.0f}".rjust(left) + "/" + f"{d:.0f}".ljust(right)
            if len(t) > width + 1 or (v != 0 and n == 0):
                t = ("*".rjust(left) + " " * (right + 1))  # too small or too large to show in this width
        cells.append(t)
    rows = x.shape[0] if x.size else 0
    cols = x.shape[1] if x.size else 0
    lines = ["".join(cells[r * cols:(r + 1) * cols]) for r in range(rows)]
    if not lines:
        return char("")
    return np.array([list(ln) for ln in lines], dtype="<U1").reshape(rows, -1)


# -- profiler --------------------------------------------------------------------------------------

class _Profiler:
    """Times user functions and scripts statement by statement while ``profile on`` is in effect."""

    def __init__(self, interp):
        self.interp = interp
        self.on = False
        self.prev = None  # the statement hook that was installed before the profiler's
        self.clear()

    def clear(self) -> None:
        self.table: dict[str, dict] = {}
        self.last: tuple | None = None  # (time, names on the call stack) at the previous statement
        self.run = object()
        self.history: list = []

    def _entry(self, name: str, frame) -> dict:
        e = self.table.get(name)
        if e is None:
            e = self.table[name] = {"calls": 0, "self": 0.0, "total": 0.0, "file": frame.file or "",
                                    "type": "M-script" if frame is self.interp.base else "M-function",
                                    "lines": {}, "children": {}}
        return e

    def _name(self, frame) -> str:
        if frame is self.interp.base:
            return os.path.splitext(os.path.basename(frame.file))[0] if frame.file else "base"
        return frame.name

    def hook(self, stmt, frame) -> None:
        now = time.perf_counter()
        if self.last is not None:
            dt = now - self.last[0]
            names = self.last[1]
            self.table[names[-1]]["self"] += dt
            for n in set(names):
                self.table[n]["total"] += dt
            line = self.last[2]
            lines = self.table[names[-1]]["lines"]
            rec = lines.setdefault(line, [0, 0.0])
            rec[0] += 1
            rec[1] += dt
        stack = tuple(self._name(f) for f in self.interp.frames)
        for f in self.interp.frames:
            self._entry(self._name(f), f)
        if getattr(frame, "_profile_run", None) is not self.run:
            frame._profile_run = self.run
            self.table[stack[-1]]["calls"] += 1
            if len(stack) > 1:
                kids = self.table[stack[-2]]["children"]
                kids[stack[-1]] = kids.get(stack[-1], 0) + 1
        self.last = (time.perf_counter(), stack, stmt.line)
        if self.prev is not None:
            self.prev(stmt, frame)

    def start(self, resume: bool) -> None:
        if self.on:
            return
        if not resume:
            self.clear()
        self.prev = self.interp.trace_hook
        self.interp.trace_hook = self.hook
        self.last = None
        self.on = True

    def stop(self) -> None:
        if not self.on:
            return
        if self.last is not None:  # time of the statement that called profile off
            self.hook_final()
        if self.interp.trace_hook == self.hook:
            self.interp.trace_hook = self.prev
        self.on = False
        self.last = None

    def hook_final(self) -> None:
        now = time.perf_counter()
        dt = now - self.last[0]
        names = self.last[1]
        self.table[names[-1]]["self"] += dt
        for n in set(names):
            self.table[n]["total"] += dt

    def info(self) -> Struct:
        names = sorted(self.table, key=lambda n: -self.table[n]["total"])
        index = {n: k + 1 for k, n in enumerate(names)}
        fields = ["CompleteName", "FunctionName", "FileName", "Type", "Children", "Parents", "ExecutedLines",
                  "IsRecursive", "TotalRecursiveTime", "PartialData", "NumCalls", "TotalTime"]
        data = np.empty((len(names), 1), dtype=object)
        for k, n in enumerate(names):
            e = self.table[n]
            kids = np.empty((len(e["children"]), 1), dtype=object)
            for j, (c, cnt) in enumerate(sorted(e["children"].items())):
                kids[j, 0] = {"Index": fscalar(index[c]), "NumCalls": fscalar(cnt),
                              "TotalTime": fscalar(self.table[c]["total"])}
            parents = [(p, self.table[p]["children"][n]) for p in names if n in self.table[p]["children"]]
            pars = np.empty((len(parents), 1), dtype=object)
            for j, (p, cnt) in enumerate(parents):
                pars[j, 0] = {"Index": fscalar(index[p]), "NumCalls": fscalar(cnt)}
            lines = np.array([[ln, c, t] for ln, (c, t) in sorted(e["lines"].items())], dtype=float).reshape(-1, 3)
            complete = f"{e['file']}>{n}" if e["file"] else n
            data[k, 0] = {"CompleteName": char(complete), "FunctionName": char(n), "FileName": char(e["file"]),
                          "Type": char(e["type"]),
                          "Children": Struct(kids, ["Index", "NumCalls", "TotalTime"]),
                          "Parents": Struct(pars, ["Index", "NumCalls"]), "ExecutedLines": lines,
                          "IsRecursive": bool_scalar(n in e["children"]), "TotalRecursiveTime": fscalar(0),
                          "PartialData": bool_scalar(False), "NumCalls": fscalar(e["calls"]),
                          "TotalTime": fscalar(e["total"])}
        return Struct.scalar({"FunctionTable": Struct(data, fields), "FunctionHistory": np.zeros((2, 0)),
                              "ClockPrecision": fscalar(time.get_clock_info("perf_counter").resolution),
                              "ClockSpeed": fscalar(0), "Name": char("mylab"), "Overhead": fscalar(0)})

    def report(self) -> str:
        names = sorted(self.table, key=lambda n: -self.table[n]["total"])
        if not names:
            return "The profiler has no data. Run code between 'profile on' and 'profile off'.\n"
        w = max(13, max(len(n) for n in names))
        lines = [f"{'Function Name':<{w}}  {'Calls':>7}  {'Total Time':>11}  {'Self Time':>10}"]
        for n in names:
            e = self.table[n]
            lines.append(f"{n:<{w}}  {e['calls']:>7}  {e['total']:>9.3f} s  {e['self']:>8.3f} s")
        return "\n".join(lines) + "\n"


def _profiler(interp) -> _Profiler:
    p = interp.state.get("profiler")
    if p is None:
        p = interp.state["profiler"] = _Profiler(interp)
    return p


@builtin("profile")
def _profile(interp, args, nargout):
    words = [text(a, "profile option") for a in args]
    action = words[0].lower() if words else "status"
    p = _profiler(interp)
    if action in ("on", "resume"):
        for w in words[1:]:
            if w.lower() not in ("-history", "-nohistory", "-historysize", "-timer", "-timestamp", "real", "cpu",
                                 "performance", "processor", "-detail", "mmex", "builtin") and not w.isdigit():
                raise MatlabError(f"profile: unknown option '{w}'.")
        p.start(resume=action == "resume")
        return None
    if action == "off":
        p.stop()
        return None
    if action == "clear":
        p.clear()
        return None
    if action == "info":
        return p.info()
    if action == "status":
        return Struct.scalar({"ProfilerStatus": char("on" if p.on else "off"), "DetailLevel": char("mmex"),
                              "Timer": char("performance"), "HistoryTracking": char("off"),
                              "HistorySize": fscalar(5000000)})
    if action in ("viewer", "report"):
        p.stop()
        interp.out(p.report())
        return None
    raise MatlabError(f"profile: unknown option '{words[0]}'. Use on, off, resume, clear, viewer, info or status.")


@builtin("profsave")
def _profsave(interp, args, nargout):
    p = _profiler(interp)
    folder = text(args[1], "folder") if len(args) > 1 else "profile_results"
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "profile.txt"), "w", encoding="utf-8") as fh:
        fh.write(p.report())
    return None


# -- futures (parfeval) ------------------------------------------------------------------------------

class Future(MObject):
    """The result of parfeval. There are no background workers: the function runs when it is queued."""

    is_handle = True
    _METHODS = {"fetchOutputs", "wait", "cancel", "fetchNext", "afterEach", "afterAll"}
    _count = 0

    def __init__(self, interp, fn: Any, nout: int, args: list):
        Future._count += 1
        self.id = Future._count
        self.fn = fn
        self.nout = nout
        self.args = list(args)
        self.outputs: list = []
        self.error: MatlabError | None = None
        self.diary: list[str] = []
        self.read = False
        self.cancelled = False
        self.started = time.time()
        old = interp.out
        interp.out = self.diary.append
        try:
            self.outputs = list(interp.call_function_value(fn, list(args), nout))
            if len(self.outputs) < nout:
                raise MatlabError(f"The function returned {len(self.outputs)} outputs but {nout} were asked for.")
        except MatlabError as e:
            self.error = e
        finally:
            interp.out = old
        self.finished = time.time()

    def class_name(self) -> str:
        return "parallel.FevalFuture"

    def isa_class(self, name: str) -> bool:
        return name in ("parallel.FevalFuture", "parallel.Future", "handle")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def fetch(self) -> list:
        if self.cancelled:
            raise MatlabError("The future was cancelled, so it has no outputs.", "parallel:fevalqueue:FetchAfterCancel")
        if self.error is not None:
            raise MatlabError(f"One or more futures resulted in an error.\n{self.error.message}",
                              self.error.identifier or "parallel:fevalqueue:ErrorFetchingOutputs")
        self.read = True
        return list(self.outputs[:self.nout])

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "fetchOutputs":
            outs = self.fetch()
            if nargout > len(outs):
                raise MatlabError(f"The future has {len(outs)} outputs but {nargout} were asked for.")
            return outs[:max(nargout, 1)] if outs else []
        if name == "wait":
            return [bool_scalar(True)] if nargout else []
        if name == "cancel":
            return []  # already finished: cancelling has no effect, as for a finished future in MATLAB
        if name == "fetchNext":
            if self.read:
                raise MatlabError("There are no unread futures to fetch.", "parallel:fevalqueue:NoUnreadFutures")
            outs = self.fetch()
            return [fscalar(1), *outs][:max(nargout, 1)]
        if name in ("afterEach", "afterAll"):
            rest = [a for a in args if a is not self]
            if len(rest) < 2:
                raise MatlabError(f"{name}(future, fcn, nout): a function and its number of outputs are needed.")
            nout = to_int(rest[1], "the number of outputs")
            return [Future(interp, rest[0], nout, self.fetch())]
        return super().call_method(interp, name, args, nargout)

    def get_field(self, interp, name: str) -> Any:
        import datetime as _dt
        values = {
            "ID": lambda: fscalar(self.id), "Function": lambda: self.fn,
            "InputArguments": lambda: Cell.from_list(self.args),
            "NumOutputArguments": lambda: fscalar(self.nout),
            "OutputArguments": lambda: Cell.from_list(self.outputs[:self.nout]),
            "State": lambda: char("finished"),
            "Read": lambda: bool_scalar(self.read),
            "Diary": lambda: char("".join(self.diary)),
            "Error": lambda: _exception(self.error),
            "RunningDuration": lambda: fscalar(self.finished - self.started),
            "CreateDateTime": lambda: char(_dt.datetime.fromtimestamp(self.started).strftime("%d-%b-%Y %H:%M:%S")),
            "StartDateTime": lambda: char(_dt.datetime.fromtimestamp(self.started).strftime("%d-%b-%Y %H:%M:%S")),
            "FinishDateTime": lambda: char(_dt.datetime.fromtimestamp(self.finished).strftime("%d-%b-%Y %H:%M:%S")),
        }
        if name in values:
            return values[name]()
        raise MatlabError(f"Unrecognized property '{name}' for class 'parallel.FevalFuture'.")

    def display_lines(self) -> list[str]:
        return [" FevalFuture with properties:", "",
                f"                   ID: {self.id}",
                f"             Function: {_fn_text(self.fn)}",
                "                State: finished " + ("(unread)" if not self.read else "(read)"),
                f"                Error: {'none' if self.error is None else self.error.message}"]


def _fn_text(f: Any) -> str:
    from ..display import function_handle_text
    return function_handle_text(f) if isinstance(f, FunctionHandle) else class_name(f)


def _exception(e: MatlabError | None) -> Any:
    if e is None:
        return np.zeros((0, 0))
    from ..objects import MExceptionObject
    return MExceptionObject(e.identifier or "", e.message)


def _is_pool(v: Any) -> bool:
    return isinstance(v, Struct) and "NumWorkers" in v.fields


@builtin("parfeval", "parfevalOnAll")
def _parfeval(interp, args, nargout):
    rest = list(args)
    if rest and _is_pool(rest[0]):
        rest = rest[1:]
    if len(rest) < 2:
        raise MatlabError("parfeval(fcn, nout, in1, ...): a function and its number of outputs are needed.")
    fn = rest[0]
    if not isinstance(fn, FunctionHandle) and not hasattr(fn, "call_object"):
        raise MatlabError("parfeval: the function must be a function handle.")
    nout = to_int(rest[1], "the number of outputs")
    if nout < 0:
        raise MatlabError("parfeval: the number of outputs must not be negative.")
    return Future(interp, fn, nout, rest[2:])


@builtin("backgroundPool")
def _background_pool(interp, args, nargout):
    return Struct.scalar({"NumWorkers": 1.0, "Busy": bool_scalar(False), "FevalQueue": Cell.empty(0, 0)})


def _future_only(name: str, example: str):
    def fn(interp, args, nargout):
        raise MatlabError(f"{name} needs a future returned by parfeval. Example: {example}")
    builtin(name)(fn)


_future_only("fetchOutputs", "f = parfeval(@max, 1, [1 5 2]); m = fetchOutputs(f)")
_future_only("fetchNext", "f = parfeval(@max, 1, [1 5 2]); [idx, m] = fetchNext(f)")


# -- Java and MEX: not available -------------------------------------------------------------------

_NO_JAVA = ("Java is not available: mylab runs on Python and has no Java virtual machine, so {what} cannot be "
            "used. Use MATLAB-language types instead (string, containers.Map, cell arrays, datetime), or call "
            "a Python library with py.name(...).")


def _no_java(what: str) -> MatlabError:
    return MatlabError(_NO_JAVA.format(what=what), "MATLAB:Java:NotAvailable")


@builtin("java.*", "javax.*")
def _java_package(interp, args, nargout):
    raise _no_java(f"the Java class {args[0]}")


def _java_function(name: str):
    def fn(interp, args, nargout):
        target = f" ('{text(args[0])}')" if args and is_text(args[0]) else ""
        raise _no_java(f"{name}{target}")
    builtin(name)(fn)


for _name in ("javaObject", "javaObjectEDT", "javaMethod", "javaMethodEDT", "javaArray", "javaaddpath",
              "javarmpath", "javaclasspath"):
    _java_function(_name)


@builtin("isjava")
def _isjava(interp, args, nargout):
    return bool_scalar(False)


# -- Python ----------------------------------------------------------------------------------------

@builtin("py.*")
def _py(interp, args, nargout):
    """py.name(...): args[0] is the full dotted name, the rest are the call's arguments."""
    from . import _pyinterop
    result = _pyinterop.dispatch(interp, args[0], list(args[1:]), nargout)
    if nargout == 0 and isinstance(result, _pyinterop.PyObject) and result.obj is None:
        return None  # a call for its effect, such as py.print(...)
    return result


@builtin("pyargs")
def _pyargs(interp, args, nargout):
    from . import _pyinterop
    if len(args) % 2:
        raise MatlabError("pyargs: arguments must be name-value pairs, for example pyargs('sep', ',').")
    pairs = {}
    for k in range(0, len(args), 2):
        key = text(args[k], "the keyword name")
        if not key.isidentifier():
            raise MatlabError(f"pyargs: '{key}' is not a valid Python keyword name.")
        pairs[key] = args[k + 1]
    return _pyinterop.PyKeywords(pairs)


@builtin("pyrun")
def _pyrun(interp, args, nargout):
    from . import _pyinterop
    return _pyinterop.pyrun(interp, list(args), nargout)


@builtin("pyrunfile")
def _pyrunfile(interp, args, nargout):
    from . import _pyinterop
    return _pyinterop.pyrunfile(interp, list(args), nargout)


@builtin("pyenv")
def _pyenv(interp, args, nargout):
    from . import _pyinterop
    _pyinterop.check_enabled(interp)
    if args:
        raise MatlabError("pyenv: the Python environment cannot be changed. mylab uses the Python it runs on; "
                          "start mylab with another Python to use a different one.")
    return _pyinterop.PyEnvironment()
