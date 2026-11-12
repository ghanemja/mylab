"""Language-support objects: onCleanup, inputParser, RandStream, timer, message.

Also small environment functions that scripts call but that have nothing to do
here (uistack, web, feature, movegui), and function-form versions of object
methods (isKey(map, k), getReport(err), parse(p, ...), start(t), ...).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MExceptionObject, MObject
from ..values import Cell, FunctionHandle, MString, Struct, char, to_str
from ._registry import builtin
from ._util import bool_scalar, fscalar, is_text, num, text


def _flag(v: Any) -> bool:
    if is_text(v):
        return text(v).lower() in ("on", "true", "yes")
    return bool(np.asarray(v).flat[0])


# -- onCleanup ----------------------------------------------------------------------------------

class OnCleanup(MObject):
    """Runs a function when the variable holding it is cleared or its function returns."""

    is_handle = True
    is_cleanup = True

    def __init__(self, task: Any):
        self.task = task
        self.done = False

    def class_name(self) -> str:
        return "onCleanup"

    def run(self, interp) -> None:
        if self.done:
            return
        self.done = True
        interp.cleanup_count = max(0, interp.cleanup_count - 1)
        try:
            interp.call_function_value(self.task, [], 0)
        except MatlabError as e:
            interp.warn(f"The onCleanup function failed: {e.message}")

    def get_field(self, interp, name: str) -> Any:
        if name == "task":
            return self.task
        raise MatlabError(f"Unrecognized property '{name}' for class 'onCleanup'.")

    def display_lines(self) -> list[str]:
        from ..display import function_handle_text
        shown = function_handle_text(self.task) if isinstance(self.task, FunctionHandle) else str(self.task)
        return ["  onCleanup with properties:", "", f"    task: {shown}"]


@builtin("onCleanup")
def _on_cleanup(interp, args, nargout):
    if len(args) != 1 or not isinstance(args[0], FunctionHandle):
        raise MatlabError("onCleanup needs a function handle with no inputs, e.g. onCleanup(@() fclose(fid)).")
    interp.cleanup_count += 1
    return OnCleanup(args[0])


# -- inputParser -----------------------------------------------------------------------------------

class InputParser(MObject):
    """inputParser: declares the arguments a function takes, then checks and sorts them."""

    is_handle = True
    _METHODS = {"addRequired", "addOptional", "addParameter", "addParamValue", "addSwitch", "parse",
                "createCopy", "disp"}
    _SETTINGS = {"FunctionName": "", "CaseSensitive": False, "KeepUnmatched": False,
                 "PartialMatching": True, "StructExpand": True}

    def __init__(self):
        self.required: list[tuple[str, Any]] = []
        self.optional: list[tuple[str, Any, Any]] = []
        self.params: dict[str, tuple[Any, Any]] = {}
        self.switches: list[str] = []
        self.settings: dict[str, Any] = dict(self._SETTINGS)
        self.results: dict[str, Any] = {}
        self.unmatched: dict[str, Any] = {}
        self.using_defaults: list[str] = []

    def class_name(self) -> str:
        return "inputParser"

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def _names(self) -> list[str]:
        return [n for n, _ in self.required] + [n for n, _, _ in self.optional] + list(self.params) \
            + self.switches

    def _check_new(self, name: str) -> None:
        if not name.isidentifier():
            raise MatlabError(f"'{name}' is not a valid argument name (it becomes a field of Results).")
        if name in self._names():
            raise MatlabError(f"An argument named '{name}' is already defined in this inputParser.")

    def get_field(self, interp, name: str) -> Any:
        if name == "Results":
            return Struct.scalar({k: self.results[k] for k in sorted(self.results, key=str.lower)})
        if name == "Unmatched":
            return Struct.scalar(dict(self.unmatched))
        if name == "UsingDefaults":
            return Cell.from_list([char(n) for n in self.using_defaults])
        if name == "Parameters":
            return Cell.from_list([char(n) for n in sorted(self._names(), key=str.lower)])
        if name == "FunctionName":
            return char(self.settings[name])
        if name in self.settings:
            return bool_scalar(self.settings[name])
        raise MatlabError(f"Unrecognized property '{name}' for class 'inputParser'.")

    def set_field(self, interp, name: str, value: Any) -> "InputParser":
        if name == "FunctionName":
            self.settings[name] = text(value)
        elif name in self.settings:
            self.settings[name] = _flag(value)
        else:
            raise MatlabError(f"Cannot set property '{name}' of an inputParser.")
        return self

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = list(args[1:])
        if name == "parse":
            self.parse(interp, rest)
            return []
        if name == "disp":
            interp.out("\n".join(self.display_lines()) + "\n")
            return []
        if name == "createCopy":
            other = InputParser()
            other.required, other.optional = list(self.required), list(self.optional)
            other.params, other.switches = dict(self.params), list(self.switches)
            other.settings = dict(self.settings)
            return [other]
        if not rest:
            raise MatlabError(f"{name} needs the argument name.")
        arg_name = text(rest[0], "argument name")
        self._check_new(arg_name)
        if name == "addRequired":
            self.required.append((arg_name, rest[1] if len(rest) > 1 else None))
        elif name == "addSwitch":
            self.switches.append(arg_name)
        else:
            if len(rest) < 2:
                raise MatlabError(f"{name} needs a default value: {name}(p, '{arg_name}', defaultValue).")
            validator = rest[2] if len(rest) > 2 and isinstance(rest[2], FunctionHandle) else None
            if name == "addOptional":
                self.optional.append((arg_name, rest[1], validator))
            else:
                self.params[arg_name] = (rest[1], validator)
        return []

    # -- parsing ------------------------------------------------------------------------------
    def _prefix(self) -> str:
        fn = self.settings["FunctionName"]
        return f"{fn}: " if fn else ""

    def _validate(self, interp, name: str, value: Any, validator: Any) -> None:
        if validator is None:
            return
        from ..display import function_handle_text
        try:
            res = interp.call_function_value(validator, [value], 0)
        except MatlabError as e:
            raise MatlabError(f"{self._prefix()}The value of '{name}' is invalid. {e.message}",
                              "MATLAB:InputParser:ArgumentFailedValidation") from None
        if res and isinstance(res[0], np.ndarray) and res[0].dtype.kind in "bfiu" and res[0].size >= 1 \
                and not bool(np.all(res[0])):
            raise MatlabError(f"{self._prefix()}The value of '{name}' is invalid. It must satisfy the "
                              f"function: {function_handle_text(validator)}.",
                              "MATLAB:InputParser:ArgumentFailedValidation")

    def _accepts(self, interp, value: Any, validator: Any) -> bool:
        try:
            self._validate(interp, "", value, validator)
        except MatlabError:
            return False
        return True

    def _match(self, key: str, names: list[str]) -> str | None:
        fold = (lambda s: s) if self.settings["CaseSensitive"] else str.lower
        k = fold(key)
        exact = [n for n in names if fold(n) == k]
        if exact:
            return exact[0]
        if self.settings["PartialMatching"] and k:
            partial = [n for n in names if fold(n).startswith(k)]
            if len(partial) == 1:
                return partial[0]
            if len(partial) > 1:
                raise MatlabError(f"{self._prefix()}'{key}' matches multiple parameter names: "
                                  + ", ".join(partial) + ". Write more of the name.",
                                  "MATLAB:InputParser:AmbiguousParamName")
        return None

    def parse(self, interp, args: list) -> None:
        self.results, self.unmatched, self.using_defaults = {}, {}, []
        args = list(args)
        if len(args) < len(self.required):
            missing = self.required[len(args)][0]
            raise MatlabError(f"{self._prefix()}Not enough input arguments. The required argument "
                              f"'{missing}' is missing.", "MATLAB:InputParser:notEnoughInputs")
        for (name, validator), value in zip(self.required, args):
            self._validate(interp, name, value, validator)
            self.results[name] = value
        rest = args[len(self.required):]
        named = list(self.params) + self.switches
        for name, default, validator in self.optional:
            take = bool(rest)
            if take and is_text(rest[0]):
                try:
                    is_name = self._match(text(rest[0]), named) is not None
                except MatlabError:
                    is_name = True
                exact = text(rest[0]).lower() in [n.lower() for n in named] and len(rest) % 2 == 0
                if is_name and (validator is None or exact or not self._accepts(interp, rest[0], validator)):
                    take = False  # the text names a parameter: the optional arguments end here
            if take and isinstance(rest[0], Struct) and self.settings["StructExpand"] \
                    and not self._accepts(interp, rest[0], validator):
                take = False  # a struct of name-value pairs
            if take:
                self._validate(interp, name, rest[0], validator)
                self.results[name] = rest.pop(0)
            else:
                self.results[name] = default
                self.using_defaults.append(name)
        pairs: list[tuple[str, Any]] = []
        k = 0
        while k < len(rest):
            a = rest[k]
            if isinstance(a, Struct) and self.settings["StructExpand"] and a.numel() == 1:
                pairs += [(f, a.data.flat[0][f]) for f in a.fields]
                k += 1
                continue
            if not is_text(a):
                raise MatlabError(f"{self._prefix()}Expected a parameter name (text) at position "
                                  f"{len(args) - len(rest) + k + 1}, but found a value of class "
                                  f"'{_class(a)}'. Too many positional arguments?",
                                  "MATLAB:InputParser:ParamMustBeChar")
            key = text(a)
            if self._match_switch(key):
                pairs.append((key, np.array([[True]])))
                k += 1
                continue
            if k + 1 >= len(rest):
                raise MatlabError(f"{self._prefix()}No value was given for the parameter '{key}'. "
                                  "Name-value arguments come in pairs.", "MATLAB:InputParser:ParamMissingValue")
            pairs.append((key, rest[k + 1]))
            k += 2
        for key, value in pairs:
            name = self._match(key, named)
            if name is None:
                if self.settings["KeepUnmatched"]:
                    self.unmatched[key] = value
                    continue
                known = ", ".join(sorted(named)) or "(none)"
                raise MatlabError(f"{self._prefix()}'{key}' is not a recognized parameter. "
                                  f"Valid parameter names: {known}.", "MATLAB:InputParser:UnmatchedParameter")
            if name in self.params:
                self._validate(interp, name, value, self.params[name][1])
            self.results[name] = value
        for name, (default, _) in self.params.items():
            if name not in self.results:
                self.results[name] = default
                self.using_defaults.append(name)
        for name in self.switches:
            if name not in self.results:
                self.results[name] = np.array([[False]])
                self.using_defaults.append(name)

    def _match_switch(self, key: str) -> bool:
        try:
            return self._match(key, list(self.params) + self.switches) in self.switches
        except MatlabError:
            return False

    def display_lines(self) -> list[str]:
        from ..display import summary
        rows = [("FunctionName", f"'{self.settings['FunctionName']}'")]
        rows += [(k, str(int(self.settings[k]))) for k in ("CaseSensitive", "KeepUnmatched",
                                                             "PartialMatching", "StructExpand")]
        for k in ("Parameters", "Results", "Unmatched", "UsingDefaults"):
            rows.append((k, summary(self.get_field(None, k), False)))
        w = max(len(k) for k, _ in rows)
        return ["  inputParser with properties:", ""] + [f"    {k:>{w}}: {v}" for k, v in rows]


def _class(v: Any) -> str:
    from ..values import class_name
    return class_name(v)


@builtin("inputParser")
def _input_parser(interp, args, nargout):
    return InputParser()


# -- RandStream ---------------------------------------------------------------------------------

_GENERATORS = {"mt19937ar", "twister", "mcg16807", "mlfg6331_64", "mrg32k3a", "philox4x32_10",
               "threefry4x64_20", "shr3cong", "swb2712", "dsfmt19937", "simdtwister", "combrecursive",
               "multfibonacci", "philox", "threefry"}


def _seed_value(v: Any) -> int | None:
    if is_text(v):
        if text(v).lower() == "shuffle":
            return None
        raise MatlabError("The seed must be a non-negative integer or 'shuffle'.")
    seed = float(np.real(num(v).flat[0]))
    if seed < 0 or np.isnan(seed) or np.isinf(seed):
        raise MatlabError("The seed must be a non-negative number.")
    return int(seed) % (2 ** 32)  # sum(100*clock) and similar non-integer seeds are rounded down


class RandStreamObject(MObject):
    """A random number stream. All generator types use NumPy's default generator."""

    is_handle = True
    _METHODS = {"rand", "randn", "randi", "randperm", "reset"}

    def __init__(self, kind: str, seed: int | None):
        self.kind = kind
        self.seed = seed
        self.gen = np.random.default_rng(seed)

    def class_name(self) -> str:
        return "RandStream"

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = list(args[1:])
        if name == "reset":
            if rest:
                self.seed = _seed_value(rest[0])
            fresh = np.random.default_rng(self.seed)
            self.gen.bit_generator.state = fresh.bit_generator.state  # keeps the global stream linked
            return []
        saved = interp.state.get("rng")
        interp.state["rng"] = self.gen
        try:
            return interp.call_builtin(name, interp.builtins[name], rest, nargout)
        finally:
            if saved is None:
                interp.state.pop("rng", None)
            else:
                interp.state["rng"] = saved

    def get_field(self, interp, name: str) -> Any:
        if name == "Type":
            return char(self.kind)
        if name == "Seed":
            return np.array([[self.seed or 0]], dtype=np.uint32)
        if name == "NumStreams" or name == "StreamIndex":
            return fscalar(1)
        if name == "NormalTransform":
            return char("Ziggurat")
        if name in ("Antithetic", "FullPrecision"):
            return bool_scalar(name == "FullPrecision")
        if name == "State":
            return _encode_state(self.gen)
        if name in self._METHODS:
            return super().get_field(interp, name)
        raise MatlabError(f"Unrecognized property '{name}' for class 'RandStream'.")

    def set_field(self, interp, name: str, value: Any) -> "RandStreamObject":
        if name == "State":
            _decode_state(self.gen, value)
        elif name not in ("NormalTransform", "Antithetic", "FullPrecision"):
            raise MatlabError(f"Cannot set property '{name}' of a RandStream.")
        return self

    def display_lines(self) -> list[str]:
        return [f"{self.kind} random stream", f"             Seed: {self.seed or 0}",
                "  NormalTransform: Ziggurat"]

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


def _encode_state(gen: np.random.Generator) -> np.ndarray:
    """The generator state as a column of uint32 words (what rng().State and s.State hold)."""
    st = gen.bit_generator.state
    words = []
    for value in (st["state"]["state"], st["state"]["inc"]):
        words += [(value >> (32 * k)) & 0xFFFFFFFF for k in range(4)]
    words += [int(st["has_uint32"]), int(st["uinteger"])]
    return np.array(words, dtype=np.uint32).reshape(-1, 1)


def _decode_state(gen: np.random.Generator, value: Any) -> None:
    w = [int(x) for x in np.asarray(value).flatten()]
    if len(w) != 10:
        raise MatlabError("This is not a generator state saved by rng or RandStream in mylab.")
    st = gen.bit_generator.state
    st["state"]["state"] = sum(w[k] << (32 * k) for k in range(4))
    st["state"]["inc"] = sum(w[4 + k] << (32 * k) for k in range(4))
    st["has_uint32"], st["uinteger"] = w[8], w[9]
    gen.bit_generator.state = st


def _new_stream(args: list) -> RandStreamObject:
    kind = text(args[0], "generator type") if args else "mt19937ar"
    if kind.lower() not in _GENERATORS:
        raise MatlabError(f"Unknown random number generator type '{kind}'. Use 'mt19937ar' (the default).")
    seed: int | None = 0
    k = 1
    while k + 1 < len(args):
        key = text(args[k]).lower()
        if key == "seed":
            seed = _seed_value(args[k + 1])
        elif key not in ("normaltransform", "numstreams", "streamindices", "cellout"):
            raise MatlabError(f"Unknown RandStream option '{text(args[k])}'.")
        k += 2
    return RandStreamObject(kind, seed)


def _global_stream(interp) -> RandStreamObject:
    stream = interp.state.get("rand_stream")
    from .core import _rng
    gen = _rng(interp)
    if stream is None or stream.gen is not gen:
        stream = RandStreamObject("mt19937ar", interp.state.get("rng_seed", 0))
        stream.gen = gen
        interp.state["rand_stream"] = stream
    return stream


@builtin("RandStream", "RandStream.create")
def _rand_stream(interp, args, nargout):
    return _new_stream(args)


@builtin("RandStream.setGlobalStream", "RandStream.setDefaultStream")
def _set_global_stream(interp, args, nargout):
    if not args or not isinstance(args[0], RandStreamObject):
        raise MatlabError("RandStream.setGlobalStream needs a stream, e.g. RandStream('mt19937ar', 'Seed', 1).")
    old = _global_stream(interp)
    interp.state["rng"] = args[0].gen
    interp.state["rand_stream"] = args[0]
    interp.state["rng_seed"] = args[0].seed or 0
    return old if nargout else None


@builtin("RandStream.getGlobalStream", "RandStream.getDefaultStream")
def _get_global_stream(interp, args, nargout):
    return _global_stream(interp)


@builtin("RandStream.list")
def _rand_stream_list(interp, args, nargout):
    names = ["mt19937ar", "dsfmt19937", "mcg16807", "mlfg6331_64", "mrg32k3a", "philox4x32_10",
             "threefry4x64_20", "shr3cong", "swb2712"]
    if nargout:
        return Cell.from_list([char(n) for n in names], column=True)
    interp.out("\nThe following random number generator algorithms are accepted (all use the same "
               "generator in mylab):\n\n" + "".join(f"    {n}\n" for n in names) + "\n")
    return None


@builtin("rng")
def _rng_control(interp, args, nargout):
    """rng(seed), rng('default'), rng('shuffle'), s = rng, rng(s)."""
    from .core import _rng
    before = None
    if nargout:
        gen = _rng(interp)
        before = Struct.scalar({"Type": char(interp.state.get("rng_type", "twister")),
                                "Seed": np.array([[interp.state.get("rng_seed", 0)]], dtype=np.uint32),
                                "State": _encode_state(gen)})
    if args:
        a = args[0]
        if isinstance(a, Struct):
            s = a.data.flat[0]
            if "State" not in s:
                raise MatlabError("rng(s) needs the settings structure returned by s = rng.")
            gen = np.random.default_rng(0)
            _decode_state(gen, s["State"])
            interp.state["rng"] = gen
            interp.state["rng_seed"] = int(np.asarray(s.get("Seed", 0)).flat[0])
        elif is_text(a):
            word = text(a).lower()
            if word == "default":
                seed = 0
            elif word == "shuffle":
                seed = int(np.random.SeedSequence().entropy % (2 ** 32))
            else:
                raise MatlabError(f"rng: unknown option '{text(a)}'. Use a seed number, 'default' or 'shuffle'.")
            interp.state["rng"] = np.random.default_rng(seed)
            interp.state["rng_seed"] = seed
        else:
            seed = _seed_value(a)
            interp.state["rng"] = np.random.default_rng(seed)
            interp.state["rng_seed"] = seed
        if len(args) > 1 and is_text(args[1]):
            kind = text(args[1])
            if kind.lower() not in _GENERATORS:
                raise MatlabError(f"rng: unknown generator type '{kind}'.")
            interp.state["rng_type"] = kind
    return before


# -- timer -----------------------------------------------------------------------------------------

class TimerObject(MObject):
    """timer, emulated without a background thread: callbacks run inside start() and wait()."""

    is_handle = True
    _METHODS = {"start", "stop", "wait", "startat", "get", "set", "isvalid"}
    MAX_RUNS = 100000

    def __init__(self):
        self.props: dict[str, Any] = {
            "AveragePeriod": fscalar(np.nan), "BusyMode": char("drop"), "ErrorFcn": char(""),
            "ExecutionMode": char("singleShot"), "InstantPeriod": fscalar(np.nan), "Name": char("timer-1"),
            "ObjectVisibility": char("on"), "Period": fscalar(1), "Running": char("off"),
            "StartDelay": fscalar(0), "StartFcn": char(""), "StopFcn": char(""), "Tag": char(""),
            "TasksExecuted": fscalar(0), "TasksToExecute": fscalar(np.inf), "TimerFcn": char(""),
            "Type": char("timer"), "UserData": np.zeros((0, 0)),
        }
        self.stop_requested = False
        self.active = False

    def class_name(self) -> str:
        return "timer"

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def _key(self, name: str) -> str:
        for k in self.props:
            if k.lower() == name.lower():
                return k
        raise MatlabError(f"Unrecognized property '{name}' for class 'timer'.")

    def get_field(self, interp, name: str) -> Any:
        return self.props[self._key(name)]

    def set_field(self, interp, name: str, value: Any) -> "TimerObject":
        key = self._key(name)
        if key in ("Running", "TasksExecuted", "AveragePeriod", "InstantPeriod", "Type"):
            raise MatlabError(f"The timer property '{key}' is read-only.")
        if key == "ExecutionMode" and text(value) not in ("singleShot", "fixedRate", "fixedDelay",
                                                           "fixedSpacing"):
            raise MatlabError("ExecutionMode must be 'singleShot', 'fixedRate', 'fixedDelay' or "
                              "'fixedSpacing'.")
        self.props[key] = value
        return self

    def _callback(self, interp, which: str, event: str) -> None:
        cb = self.props[which]
        if isinstance(cb, np.ndarray) and cb.size == 0:
            return
        if is_text(cb):
            interp.call_builtin("eval", interp.builtins["eval"], [char(text(cb))], 0)
            return
        data = Struct.scalar({"Type": char(event),
                              "Data": Struct.scalar({"time": interp.call_builtin(
                                  "clock", interp.builtins["clock"], [], 1)[0]})})
        if isinstance(cb, Cell):
            items = cb.values_f()
            interp.call_function_value(items[0], [self, data] + items[1:], 0)
        else:
            interp.call_function_value(cb, [self, data], 0)

    def _tick(self, interp) -> None:
        try:
            self._callback(interp, "TimerFcn", "TimerFcn")
        except MatlabError:
            self._finish(interp, error=True)
            raise
        self.props["TasksExecuted"] = fscalar(float(self.props["TasksExecuted"].flat[0]) + 1)

    def _limit(self) -> float:
        if text(self.props["ExecutionMode"]) == "singleShot":
            return 1
        return float(np.asarray(self.props["TasksToExecute"]).flat[0])

    def _finish(self, interp, error: bool = False) -> None:
        if not self.active:
            return
        self.active = False
        self.props["Running"] = char("off")
        if error:
            try:
                self._callback(interp, "ErrorFcn", "ErrorFcn")
            except MatlabError:
                pass
        self._callback(interp, "StopFcn", "StopFcn")

    def start(self, interp) -> None:
        if getattr(self, "_deleted", False):
            raise MatlabError("Invalid or deleted timer object.")
        if self.active:
            raise MatlabError("The timer is already running; stop it first.")
        self.active, self.stop_requested = True, False
        self.props["Running"] = char("on")
        self.props["TasksExecuted"] = fscalar(0)
        self._callback(interp, "StartFcn", "StartFcn")
        limit = self._limit()
        if np.isinf(limit):
            self._run(interp, 1)  # an endless timer runs once now; wait(t) keeps it going
        else:
            self._run(interp, int(limit))
            self._finish(interp)

    def _run(self, interp, count: int) -> None:
        for _ in range(count):
            if self.stop_requested or not self.active:
                break
            self._tick(interp)

    def stop(self, interp) -> None:
        self.stop_requested = True
        self._finish(interp)

    def wait(self, interp) -> None:
        runs = 0
        while self.active and not self.stop_requested:
            self._tick(interp)
            runs += 1
            if runs >= self.MAX_RUNS:
                interp.warn(f"wait: the timer callback ran {runs} times without calling stop; stopping it.")
                break
        self._finish(interp)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = list(args[1:])
        if name in ("start", "startat"):
            self.start(interp)
        elif name == "stop":
            self.stop(interp)
        elif name == "wait":
            self.wait(interp)
        elif name == "isvalid":
            return [bool_scalar(not getattr(self, "_deleted", False))]
        elif name == "get":
            if rest:
                return [self.get_field(interp, text(rest[0]))]
            return [Struct.scalar(dict(self.props))]
        elif name == "set":
            for k in range(0, len(rest) - 1, 2):
                self.set_field(interp, text(rest[k]), rest[k + 1])
        return []

    def display_lines(self) -> list[str]:
        p = self.props
        mode = text(p["ExecutionMode"])
        return [f"   Timer Object: {text(p['Name'])}", "",
                "   Timer Settings", f"      ExecutionMode: {mode}",
                f"             Period: {float(p['Period'].flat[0]):g}",
                f"           BusyMode: {text(p['BusyMode'])}", f"            Running: {text(p['Running'])}"]


@builtin("timer")
def _timer(interp, args, nargout):
    t = TimerObject()
    count = interp.state.get("timer_count", 0) + 1
    interp.state["timer_count"] = count
    t.props["Name"] = char(f"timer-{count}")
    if len(args) % 2:
        raise MatlabError("timer options come in name-value pairs, e.g. timer('TimerFcn', @(~,~) disp('tick')).")
    for k in range(0, len(args), 2):
        t.set_field(interp, text(args[k], "property name"), args[k + 1])
    return t


@builtin("timerfind", "timerfindall")
def _timerfind(interp, args, nargout):
    return np.zeros((0, 0))


# -- message ------------------------------------------------------------------------------------------

class MessageObject(MObject):
    """message(id, args...): an entry of a message catalog. mylab has no catalog, so the text is the id."""

    is_message = True
    _METHODS = {"getString", "string", "char"}

    def __init__(self, identifier: str, arguments: list):
        self.identifier = identifier
        self.arguments = arguments

    def class_name(self) -> str:
        return "message"

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def text(self) -> str:
        from ..display import summary
        if not self.arguments:
            return self.identifier
        shown = [to_str(a) if is_text(a) else summary(a, False) for a in self.arguments]
        return f"{self.identifier} ({', '.join(shown)})"

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "string":
            return [MString(self.text())]
        return [char(self.text())]

    def get_field(self, interp, name: str) -> Any:
        if name == "Identifier":
            return char(self.identifier)
        if name == "Arguments":
            return Cell.from_list(self.arguments)
        return super().get_field(interp, name)

    def display_lines(self) -> list[str]:
        return ["  message with properties:", "", f"    Identifier: '{self.identifier}'",
                f"     Arguments: {{1×{len(self.arguments)} cell}}"]


@builtin("message")
def _message(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("message needs a message identifier, e.g. message('MyPkg:myId').")
    return MessageObject(text(args[0]), list(args[1:]))


# -- harmless environment functions ---------------------------------------------------------------------

@builtin("uistack", "movegui", "workspace", "figurepalette", "plotbrowser", "propertyeditor", "plottools",
         "brush", "linkdata", "linkprop")
def _ignored(interp, args, nargout):
    return np.zeros((0, 0)) if nargout else None


@builtin("web")
def _web(interp, args, nargout):
    if args and not nargout:
        interp.out(f"(web: not opening a browser for {text(args[0])})\n")
    if nargout:
        return (fscalar(0), np.zeros((0, 0)), char(text(args[0]) if args else ""))[:max(nargout, 1)]
    return None


_FEATURES = {"defaultcharacterset": "UTF-8", "locale": None, "hotlinks": 0.0, "accel": 1.0, "jit": 1.0,
             "usehg2": 1.0, "isdmlworker": 0.0, "shadowfigureshandle": 0.0, "isretina": 0.0}


@builtin("feature")
def _feature(interp, args, nargout):
    if not args:
        raise MatlabError("feature needs the name of a setting, e.g. feature('numcores').")
    name = text(args[0]).lower()
    if name in ("numcores", "numthreads"):
        return fscalar(os.cpu_count() or 1)
    if name == "getpid":
        return fscalar(os.getpid())
    if name == "locale":
        info = {k: char("en_US.UTF-8") for k in ("ctype", "collate", "time", "numeric", "monetary",
                                                  "messages")}
        info["encoding"] = char("UTF-8")
        return Struct.scalar(info)
    if name == "memstats":
        return fscalar(0)
    value = _FEATURES.get(name, 0.0)
    return char(value) if isinstance(value, str) else fscalar(value)


@builtin("javachk")
def _javachk(interp, args, nargout):
    return Struct.scalar({"message": char("Java is not available in mylab."),
                          "identifier": char("MATLAB:javachk:thisFeatureNotAvailable")})


# -- function-form versions of object methods ------------------------------------------------------------

def _needs(name: str, kind: str, example: str):
    def fn(interp, args, nargout):
        got = _class(args[0]) if args else "nothing"
        raise MatlabError(f"{name} needs {kind} as its first argument (it got {got}). Example: {example}")
    builtin(name)(fn)


for _name in ("isKey", "keys", "values", "remove"):
    _needs(_name, "a containers.Map", "m = containers.Map(); m('a') = 1; isKey(m, 'a')")
for _name in ("addRequired", "addOptional", "addParameter", "addParamValue", "addSwitch", "parse"):
    _needs(_name, "an inputParser", "p = inputParser; addParameter(p, 'Tol', 1e-3); parse(p, varargin{:})")
for _name in ("start", "stop", "wait"):
    _needs(_name, "a timer", "t = timer('TimerFcn', @(~,~) disp('tick')); start(t)")
_needs("getString", "a message object", "getString(message('MyPkg:myId'))")


@builtin("getReport")
def _get_report(interp, args, nargout):
    if not args or not isinstance(args[0], MExceptionObject):
        raise MatlabError("getReport needs the error caught by try/catch: catch err, disp(getReport(err)), end")
    return args[0].call_method(interp, "getReport", args, nargout)[0]


@builtin("throwAsCaller")
def _throw_as_caller(interp, args, nargout):
    if not args or not isinstance(args[0], MExceptionObject):
        raise MatlabError("throwAsCaller needs an MException object.")
    raise args[0].to_error()


@builtin("step")
def _step(interp, args, nargout):
    from ..classdef import UserObject
    obj = args[0] if args else None
    if isinstance(obj, UserObject) and "stepImpl" in obj.cls.methods:
        if "setupImpl" in obj.cls.methods and not getattr(obj, "_setup_done", False):
            obj._setup_done = True
            setup = obj.cls.methods["setupImpl"][0]
            obj.cls.call(interp, "setupImpl", list(args[:len(setup.params)]), 0)
        return tuple(obj.cls.call(interp, "stepImpl", list(args), max(nargout, 1))) or None
    raise MatlabError("step works on System objects (classes with a stepImpl method). Step responses of "
                      "control-system models are not supported.")
