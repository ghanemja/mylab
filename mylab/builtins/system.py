"""Output, errors, workspace management, evaluation, files and timing."""

from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

from .. import __version__
from ..display import FORMAT, disp_text, format_display, function_handle_text
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, Struct, char, is_char, to_int, to_str,
)
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, is_text, nargs, num, text
from .strings import matlab_sprintf


@builtin("disp")
def _disp(interp, args, nargout):
    nargs(args, 1, 1, "disp")
    out = disp_text(args[0])
    if out:
        interp.out(out)
    elif isinstance(args[0], (np.ndarray,)) and args[0].size == 0 and is_char(args[0]):
        interp.out("\n")
    return None


@builtin("display")
def _display(interp, args, nargout):
    """display(x) prints x the way an unsuppressed statement does, under the name of the argument."""
    names = interp.state.get("call_arg_names") or []
    name = text(args[1]) if len(args) > 1 else (names[0] if names and names[0] else "ans")
    interp.out(format_display(name, args[0]))
    return None


def _error_from_args(args) -> MatlabError:
    from ..objects import MExceptionObject
    if len(args) == 1 and isinstance(args[0], MExceptionObject):
        return args[0].to_error()
    if args and getattr(args[0], "is_message", False):  # error(message('pkg:id', ...))
        return MatlabError(args[0].text(), args[0].identifier)
    if len(args) == 1 and isinstance(args[0], Struct):
        s = args[0].data.flat[0]
        return MatlabError(to_str(s.get("message", char(""))), to_str(s.get("identifier", char(""))) or "")
    first = text(args[0], "error message")
    ident = ""
    rest = list(args[1:])
    # error('pkg:id', 'message %d', x): an identifier has a colon and no spaces
    if rest and ":" in first and " " not in first and not first.endswith(":"):
        ident = first
        first = text(rest[0], "error message")
        rest = rest[1:]
    # A message given alone is printed literally; with arguments (or an identifier) it is a format.
    msg = matlab_sprintf(first, rest) if (rest or ident) else first
    return MatlabError(msg, ident)


@builtin("error")
def _error(interp, args, nargout):
    nargs(args, 1, None, "error")
    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].size == 0:
        return None
    raise _error_from_args(args)


@builtin("rethrow", "throw")
def _rethrow(interp, args, nargout):
    raise _error_from_args(args)


@builtin("MException")
def _mexception(interp, args, nargout):
    from ..objects import MExceptionObject
    ident = text(args[0])
    msg = matlab_sprintf(text(args[1]), args[2:]) if len(args) > 1 else ""
    return MExceptionObject(ident, msg)


_WARNING_MODES = ("backtrace", "verbose")


def _warning_state(interp, ident: str) -> str:
    """'on', 'off' or 'error' for one warning identifier ('all' for the default state)."""
    if ident in _WARNING_MODES:
        return interp.state.get("warning_modes", {}).get(ident, "on" if ident == "backtrace" else "off")
    if ident != "all":
        if ident in interp.state.get("warning_errors", ()):
            return "error"
        if ident in interp.disabled_warnings:
            return "off"
        if ident in interp.state.get("warnings_on", ()):
            return "on"
    return "on" if interp.warnings_enabled else "off"


def _state_struct(rows: list) -> Struct:
    data = np.empty((len(rows), 1), dtype=object)
    for k, (ident, state) in enumerate(rows):
        data[k, 0] = {"identifier": char(ident), "state": char(state)}
    return Struct(data, ["identifier", "state"])


def _all_warning_states(interp) -> Struct:
    idents = sorted(set(interp.disabled_warnings) | set(interp.state.get("warnings_on", ()))
                    | set(interp.state.get("warning_errors", ())))
    return _state_struct([("all", _warning_state(interp, "all"))] + [(i, _warning_state(interp, i)) for i in idents])


def _set_warning_state(interp, state: str, ident: str) -> None:
    on_set = interp.state.setdefault("warnings_on", set())
    errors = interp.state.setdefault("warning_errors", set())
    if ident in _WARNING_MODES:
        if state == "error":
            raise MatlabError(f"warning: '{ident}' can only be turned 'on' or 'off'.")
        interp.state.setdefault("warning_modes", {})[ident] = state
        return
    if ident == "all":
        interp.disabled_warnings.clear()
        on_set.clear()
        errors.clear()
        if state == "error":
            raise MatlabError("warning: turning every warning into an error with warning('error', 'all') is "
                              "not supported; name the warning identifier.")
        interp.warnings_enabled = state == "on"
        return
    interp.disabled_warnings.discard(ident)
    on_set.discard(ident)
    errors.discard(ident)
    if state == "off":
        interp.disabled_warnings.add(ident)
    elif state == "error":
        errors.add(ident)
    elif not interp.warnings_enabled:
        on_set.add(ident)  # on although the default is off


def _issue_warning(interp, msg: str, ident: str) -> None:
    state = _warning_state(interp, ident) if ident else _warning_state(interp, "all")
    if state == "error":
        raise MatlabError(msg, ident)
    capture = interp.__dict__.get("warn")
    if capture is not None:  # something is collecting warnings (the unit-testing framework)
        capture(msg, ident)
        return
    interp.last_warning = msg
    interp.last_warning_id = ident
    if state == "on":
        interp.out(f"Warning: {msg}\n")


@builtin("warning")
def _warning(interp, args, nargout):
    """Issue a warning, or query and change which warnings are shown (returning the old state)."""
    if not args:
        states = _all_warning_states(interp)
        if nargout:
            return states
        interp.out(f"By default, warnings are {_warning_state(interp, 'all')}.\n")
        for d in states.data.flat[1:]:
            interp.out(f"  {to_str(d['state']):>5}  {to_str(d['identifier'])}\n")
        return None
    if isinstance(args[0], Struct):
        s = args[0]
        if "state" in s.fields and "identifier" in s.fields:  # warning(w): restore saved states
            before = _all_warning_states(interp)
            for d in s.data.flatten(order="F"):
                state = to_str(d["state"]).lower()
                if state not in ("on", "off", "error"):
                    raise MatlabError("warning: the state in the structure must be 'on', 'off' or 'error'.")
                _set_warning_state(interp, state, to_str(d["identifier"]))
            return before if nargout else None
        if "message" in s.fields and s.data.size == 1:  # warning(struct with message and identifier)
            d = s.data.flat[0]
            _issue_warning(interp, to_str(d["message"]), to_str(d["identifier"]) if "identifier" in d else "")
            return None
        raise MatlabError("warning: the structure must have the fields 'identifier' and 'state' (as returned "
                          "by warning) or 'message'.")
    if getattr(args[0], "is_message", False):  # warning(message('pkg:id', ...))
        _issue_warning(interp, args[0].text(), args[0].identifier)
        return None
    first = text(args[0], "the warning message")
    low = first.lower()
    if low in ("on", "off", "error", "query") and len(args) <= 2:
        ident = text(args[1], "the warning identifier") if len(args) > 1 else "all"
        if ident == "last":
            ident = getattr(interp, "last_warning_id", "") or "all"
        if low == "query":
            if nargout:
                return _all_warning_states(interp) if ident == "all" else \
                    _state_struct([(ident, _warning_state(interp, ident))])
            if ident == "all":
                return _warning(interp, [], 0)
            interp.out(f"The state of warning '{ident}' is '{_warning_state(interp, ident)}'.\n")
            return None
        before = _all_warning_states(interp) if ident == "all" else \
            _state_struct([(ident, _warning_state(interp, ident))])
        _set_warning_state(interp, low, ident)
        return before if nargout else None
    rest = list(args[1:])
    ident = ""
    if rest and ":" in first and " " not in first and not first.endswith(":"):
        ident = first
        first = text(rest[0], "the warning message")
        rest = rest[1:]
    if first == "" and not ident:
        return None  # warning('') does nothing
    _issue_warning(interp, matlab_sprintf(first, rest) if (rest or ident) else first, ident)
    return None


@builtin("lastwarn")
def _lastwarn(interp, args, nargout):
    msg, ident = char(interp.last_warning), char(getattr(interp, "last_warning_id", ""))
    if args:
        interp.last_warning = text(args[0])
        interp.last_warning_id = text(args[1]) if len(args) > 1 else ""
    return (msg, ident) if nargout > 1 else msg


@builtin("assert")
def _assert(interp, args, nargout):
    nargs(args, 1, None, "assert")
    from ..values import is_true
    try:
        ok = is_true(args[0])
    except MatlabError:
        ok = False
    if not ok:
        if len(args) > 1:
            raise _error_from_args(args[1:])
        raise MatlabError("Assertion failed.", "MATLAB:assertion:failed")
    return None


# -- timing ----------------------------------------------------------------------------

@builtin("tic")
def _tic(interp, args, nargout):
    t = time.perf_counter()
    interp.state["tic"] = t
    if nargout > 0:
        return float(int(t * 1e6))
    return None


@builtin("toc")
def _toc(interp, args, nargout):
    if args:
        start = num(args[0]).flat[0] / 1e6
    else:
        start = interp.state.get("tic")
        if start is None:
            raise MatlabError("You must call TIC without an output argument before calling TOC "
                              "without an input argument.")
    elapsed = time.perf_counter() - start
    if nargout > 0:
        return elapsed
    interp.out(f"Elapsed time is {elapsed:.6f} seconds.\n")
    return None


@builtin("cputime")
def _cputime(interp, args, nargout):
    return time.process_time()


# -- workspace -------------------------------------------------------------------------

@builtin("clc", "home")
def _clc(interp, args, nargout):
    hook = interp.state.get("clc_hook")
    if hook:
        hook()
    return None


def _run_cleanups(interp, before: dict, after: dict) -> None:
    for n, v in before.items():
        if getattr(v, "is_cleanup", False) and after.get(n) is not v:
            v.run(interp)


@builtin("clear", "clearvars")
def _clear_with_cleanup(interp, args, nargout):
    if not interp.cleanup_count:
        return _clear(interp, args, nargout)
    frame = interp.frame
    before = dict(frame.vars)
    try:
        return _clear(interp, args, nargout)
    finally:
        _run_cleanups(interp, before, frame.vars)


def _clear(interp, args, nargout):
    frame = interp.frame
    names = [text(a) for a in args]
    if "-except" in names:
        import fnmatch
        k = names.index("-except")
        keep = names[k + 1:]
        targets = names[:k] or ["*"]
        for n in list(frame.vars):
            if any(fnmatch.fnmatchcase(n, t) for t in targets) and \
                    not any(fnmatch.fnmatchcase(n, p) for p in keep):
                del frame.vars[n]
        return None
    if not names or names in (["all"], ["variables"], ["-all"]):
        frame.vars.clear()
        if names and names[0] in ("all", "-all"):
            interp.persistent_vars.clear()
            interp.global_vars.clear()
            _clear_functions(interp, frame, None)
            if frame is interp.base:
                frame.imports = None
            from .plotting import close_all
            close_all(interp)
        return None
    if names[0] in ("global", "-global"):
        for n in names[1:] or list(interp.global_vars):
            interp.global_vars.pop(n, None)
        return None
    if names[0] in ("functions", "-functions", "classes", "-classes", "import", "-import", "mex"):
        kind = names[0].lstrip("-")
        if kind == "import":
            if frame is not interp.base:
                raise MatlabError("clear import can only be used at the command prompt.")
            frame.imports = None
            return None
        if kind == "classes":  # also clears every variable, like MATLAB
            frame.vars.clear()
            interp.classes.clear()
        _clear_functions(interp, frame, names[1:] if kind == "functions" and len(names) > 1 else None)
        return None
    import fnmatch
    for pattern in names:
        hit = False
        for n in list(frame.vars):
            if fnmatch.fnmatchcase(n, pattern):
                del frame.vars[n]
                hit = True
        frame.globals.pop(pattern, None)
        if not hit:  # clear name: a function of that name loses its persistent variables
            _clear_functions(interp, frame, [pattern])
    return None


def _clear_functions(interp, frame, patterns) -> None:
    """Forget persistent variables: of every function (``patterns`` None) or of the named ones."""
    locked = interp.state.get("mlocked", set())
    stores = interp.persistent_vars
    owners = interp.state.get("persistent_owners", {})
    nodes = interp.state.get("persistent_nodes", {})
    if patterns is None:
        for key in list(stores):
            if owners.get(key) not in locked:
                del stores[key]
                owners.pop(key, None)
                nodes.pop(key, None)
        return
    import fnmatch
    for key in list(stores):
        name = owners.get(key)
        if name is not None and name not in locked and any(fnmatch.fnmatchcase(name, p) for p in patterns):
            del stores[key]
            owners.pop(key, None)
            nodes.pop(key, None)


@builtin("who")
def _who(interp, args, nargout):
    """who, who a* b, who -regexp pat, who global, who -file name: the variable names."""
    from .fileformats import listing_rows
    names = [r["name"] for r in listing_rows(interp, args, "who")]
    if nargout > 0:
        return Cell.from_list([char(n) for n in names], column=True)
    if names:
        interp.out("\nYour variables are:\n\n" + "  ".join(names) + "\n\n")
    return None


@builtin("whos")
def _whos(interp, args, nargout):
    """whos with the same selections as who; s = whos(...) returns a struct array instead of printing."""
    from .fileformats import listing_rows, whos_struct, whos_text
    rows = listing_rows(interp, args, "whos")
    if nargout > 0:
        return whos_struct(rows)
    interp.out(whos_text(rows))
    return None


def _is_model_file(name: str) -> bool:
    """A Simulink model or library file, for which exist answers 4."""
    return os.path.splitext(name)[1].lower() in (".slx", ".mdl")


@builtin("exist")
def _exist(interp, args, nargout):
    name = text(args[0])
    kind = text(args[1]) if len(args) > 1 else ""
    frame = interp.frame
    if kind in ("", "var") and frame.has(name):
        return 1.0
    if kind == "var":
        return 0.0
    if kind in ("", "file", "dir"):
        if os.path.isdir(name):
            return 7.0
        if kind != "dir" and os.path.isfile(name):
            return 4.0 if _is_model_file(name) else 2.0
    if kind in ("", "file") and name and not os.path.isabs(name):
        for d in interp._search_dirs():  # any file on the search path, data files included (MATLAB's 2)
            if os.path.isfile(os.path.join(d, name)):
                return 4.0 if _is_model_file(name) else 2.0
    if kind in ("", "file", "builtin"):
        if interp.find_file_function(name) is not None:
            return 2.0
        if name in frame.functions:
            return 2.0
        if name in BUILTINS:
            return 5.0
    return 0.0


@builtin("isvarname")
def _isvarname(interp, args, nargout):
    import re
    from ..lexer import KEYWORDS
    s = text(args[0])
    return bool_scalar(bool(re.fullmatch(r"[A-Za-z]\w{0,62}", s)) and s not in KEYWORDS)


@builtin("iskeyword")
def _iskeyword(interp, args, nargout):
    from ..lexer import KEYWORDS
    if not args:
        return Cell.from_list([char(k) for k in sorted(KEYWORDS)], column=True)
    return bool_scalar(text(args[0]) in KEYWORDS)


def _declared_count(interp, f, outputs: bool) -> float:
    """nargin(f) / nargout(f): the declared count, negative when the list ends in varargin/varargout."""
    from ..interpreter import NestedRef
    last = "varargout" if outputs else "varargin"
    if isinstance(f, FunctionHandle) and f.is_anonymous:
        if outputs:
            return -1.0  # an anonymous function returns whatever its expression returns
        names = list(f.params)
    else:
        if isinstance(f, FunctionHandle):
            name, functions = f.name, f.closure.get("__functions__", {})
        else:
            name, functions = text(f, "function name"), interp.frame.functions
        fn = functions.get(name) or interp.frame.functions.get(name)
        if isinstance(fn, NestedRef):
            fn = fn.fn
        if fn is None:
            found = interp.find_file_function(name)
            if found is not None and found[0].functions and not found[0].body:
                fn = next(iter(found[0].functions.values()))
            elif found is not None and not found[0].classdefs:
                raise MatlabError(f"{name} is a script.", "MATLAB:nargin:isScript")
        if fn is None:
            if name in BUILTINS:
                return -1.0  # built-in functions do not declare their argument lists
            raise MatlabError(f"Function '{name}' not found.", "MATLAB:narginout:functionDoesnotExist")
        names = list(fn.outputs if outputs else fn.params)
    if names and names[-1] == last:
        return -float(len(names))
    return float(len(names))


@builtin("nargin")
def _nargin(interp, args, nargout):
    if args:
        return _declared_count(interp, args[0], False)
    return float(interp.frame.nargin)


@builtin("nargout")
def _nargout(interp, args, nargout):
    if args:
        return _declared_count(interp, args[0], True)
    return float(interp.frame.nargout)


@builtin("narginchk")
def _narginchk(interp, args, nargout):
    lo, hi = num(args[0]).flat[0], num(args[1]).flat[0]
    n = interp.frame.nargin
    if n < lo:
        raise MatlabError("Not enough input arguments.", "MATLAB:narginchk:notEnoughInputs")
    if n > hi:
        raise MatlabError("Too many input arguments.", "MATLAB:narginchk:tooManyInputs")
    return None


@builtin("nargoutchk")
def _nargoutchk(interp, args, nargout):
    return None


@builtin("exit", "quit")
def _exit(interp, args, nargout):
    code = to_int(args[0]) if args and not is_text(args[0]) else 0
    raise SystemExit(code)


# -- evaluation ---------------------------------------------------------------------------

@builtin("feval")
def _feval(interp, args, nargout):
    nargs(args, 1, None, "feval")
    res = interp.call_function_value(args[0], list(args[1:]), nargout)
    if not res:
        return None
    return tuple(res) if len(res) > 1 else res[0]


@builtin("func2str")
def _func2str(interp, args, nargout):
    f = args[0]
    if not isinstance(f, FunctionHandle):
        raise MatlabError("func2str: input must be a function handle.")
    return char(function_handle_text(f) if f.is_anonymous else str(f.name))


@builtin("str2func")
def _str2func(interp, args, nargout):
    s = text(args[0]).strip()
    if s.startswith("@"):
        return interp.eval_expression(s)
    from ..values import FunctionHandle as FH
    return FH(name=s, closure={"__functions__": interp.frame.functions})


@builtin("eval")
def _eval(interp, args, nargout):
    from ..parser import parse
    code = text(args[0])
    frame = interp.frame
    try:
        if nargout > 0:
            program = parse(code)
            if len(program.body) == 1 and type(program.body[0]).__name__ == "ExprStmt":
                vals = interp.eval_multi(program.body[0].expr, frame, nargout)
                return tuple(vals) if len(vals) > 1 else (vals[0] if vals else None)
        interp.run_program(parse(code), frame)
    except MatlabError:
        if len(args) > 1:
            interp.run_program(parse(text(args[1])), frame)
        else:
            raise
    return None


@builtin("evalc")
def _evalc(interp, args, nargout):
    from ..parser import parse
    captured: list[str] = []
    old = interp.out
    interp.out = captured.append
    extra: tuple = ()
    try:
        program = parse(text(args[0]))
        if nargout > 1:  # [text, out1, ..., outN] = evalc(expression)
            if not (len(program.body) == 1 and type(program.body[0]).__name__ == "ExprStmt"):
                raise MatlabError("evalc: outputs other than the captured text need an expression, not a statement.")
            extra = tuple(interp.eval_multi(program.body[0].expr, interp.frame, nargout - 1))
        else:
            interp.run_program(program, interp.frame)
    finally:
        interp.out = old
    return (char("".join(captured)), *extra) if extra else char("".join(captured))


@builtin("input")
def _input(interp, args, nargout):
    prompt = text(args[0]) if args else ""
    reply = interp.input_func(prompt)
    if len(args) > 1 and text(args[1]) == "s":
        return char(reply)
    if not reply.strip():
        return np.zeros((0, 0))
    return interp.eval_expression(reply)


class FormatOptions(MObject):
    """The value returned by ``fmt = format``: the numeric format and the line spacing."""

    def __init__(self, numeric: str, spacing: str):
        self.numeric = numeric
        self.spacing = spacing

    def class_name(self) -> str:
        return "matlab.internal.display.format.DisplayFormatOptions"

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "DisplayFormatOptions")

    def get_field(self, interp, name: str):
        if name == "NumericFormat":
            return MString(self.numeric)
        if name == "LineSpacing":
            return MString(self.spacing)
        raise MatlabError(f"Unrecognized property '{name}' for class 'DisplayFormatOptions'.")

    def set_field(self, interp, name: str, value):
        if name == "NumericFormat":
            probe = type(FORMAT)()
            if not probe.set_mode(text(value)):
                raise MatlabError(f"Unknown numeric format '{text(value)}'.")
            return FormatOptions(probe.numeric_name, self.spacing)
        if name == "LineSpacing":
            if text(value).lower() not in ("loose", "compact"):
                raise MatlabError("LineSpacing must be 'loose' or 'compact'.")
            return FormatOptions(self.numeric, text(value).lower())
        raise MatlabError(f"Unrecognized property '{name}' for class 'DisplayFormatOptions'.")

    def display_lines(self) -> list[str]:
        return ["  DisplayFormatOptions with properties:", "",
                f'    NumericFormat: "{self.numeric}"',
                f'      LineSpacing: "{self.spacing}"']

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


@builtin("format")
def _format(interp, args, nargout):
    """format, format short|long|shortG|..., format compact|loose, fmt = format, format(fmt)."""
    current = FormatOptions(FORMAT.numeric_name, FORMAT.spacing)
    if len(args) == 1 and isinstance(args[0], FormatOptions):
        FORMAT.set_mode(args[0].numeric)
        FORMAT.spacing = args[0].spacing
        return current if nargout else None
    if not args:
        if nargout:
            return current
        FORMAT.reset()
        return None
    words = [text(a) for a in args]
    joined = "".join(words).lower().replace(" ", "")
    if joined == "default":
        FORMAT.reset()
    elif joined in ("compact", "loose"):
        FORMAT.spacing = joined
    elif not FORMAT.set_mode(joined):
        raise MatlabError(f"Unknown command option '{' '.join(words)}'. Use short, long, shortE, longE, shortG, "
                          "longG, shortEng, longEng, bank, hex, rat, +, compact or loose.",
                          "MATLAB:format:UnknownOption")
    return current if nargout else None


@builtin("more", "beep", "commandwindow", "pack")
def _noop(interp, args, nargout):
    return None


# mylab answers version questions as this MATLAB release, so that code which checks for a
# "new enough" MATLAB takes its modern branch.
MATLAB_VERSION = "9.14.0"
MATLAB_RELEASE = "R2023a"
MATLAB_RELEASE_DATE = "February 22, 2023"
# name, folder name (as used by ver and verLessThan), version
PRODUCTS = [
    ("MATLAB", "matlab", "9.14"),
    ("Image Processing Toolbox", "images", "11.7"),
    ("Signal Processing Toolbox", "signal", "9.2"),
    ("Statistics and Machine Learning Toolbox", "stats", "12.5"),
    ("Optimization Toolbox", "optim", "9.5"),
]


@builtin("version")
def _version(interp, args, nargout):
    full = f"{MATLAB_VERSION} ({MATLAB_RELEASE}) mylab {__version__}"
    if args:
        flag = text(args[0]).lower()
        if flag == "-release":
            return char(MATLAB_RELEASE[1:])
        if flag == "-date":
            return char(MATLAB_RELEASE_DATE)
        if flag == "-description":
            return char(f"mylab {__version__}")
        if flag == "-java":
            return char("Java is not enabled")
        if flag in ("-blas", "-lapack", "-fftw", "-hdfversion"):
            return char(f"NumPy {np.__version__}")
        raise MatlabError(f"Unknown version option '{text(args[0])}'.")
    return (char(full), char(MATLAB_RELEASE_DATE)) if nargout > 1 else char(full)


@builtin("ver")
def _ver(interp, args, nargout):
    want = text(args[0]).lower() if args else None
    rows = [p for p in PRODUCTS if want is None or want in (p[1], p[0].lower())]
    if nargout > 0:
        data = np.empty((1, len(rows)), dtype=object)
        for i, (name, _folder, ver) in enumerate(rows):
            data[0, i] = {"Name": char(name), "Version": char(ver), "Release": char(f"({MATLAB_RELEASE})"),
                          "Date": char(MATLAB_RELEASE_DATE)}
        return Struct(data, ["Name", "Version", "Release", "Date"])
    lines = [f"mylab version {__version__} (Python {sys.version.split()[0]}, NumPy {np.__version__})",
             f"Reports itself to programs as MATLAB {MATLAB_VERSION} ({MATLAB_RELEASE})."]
    for name, _folder, ver in rows:
        lines.append(f"{name:<52}Version {ver:<8}({MATLAB_RELEASE})")
    if want is not None and not rows:
        lines.append(f"No product named '{text(args[0])}' is available.")
    interp.out("\n".join(lines) + "\n")
    return None


@builtin("isdeployed")
def _false_pred(interp, args, nargout):
    return bool_scalar(False)


@builtin("ismac")
def _ismac(interp, args, nargout):
    return bool_scalar(sys.platform == "darwin")


@builtin("ispc")
def _ispc(interp, args, nargout):
    return bool_scalar(os.name == "nt")


@builtin("isunix")
def _isunix(interp, args, nargout):
    return bool_scalar(os.name == "posix")


def _architecture() -> str:
    import platform
    machine = platform.machine().lower()
    if sys.platform.startswith("win"):
        return "win64"
    if sys.platform == "darwin":
        return "maca64" if machine in ("arm64", "aarch64") else "maci64"
    if sys.platform == "emscripten":
        return "wasm32"
    return "glnxa64" if machine in ("x86_64", "amd64", "") else f"glnx{machine}"


@builtin("computer")
def _computer(interp, args, nargout):
    """computer, computer('arch'), [name, maxsize, endian] = computer."""
    arch = _architecture()
    if args:
        if text(args[0]).lower() != "arch":
            raise MatlabError("computer: the only option is 'arch'.")
        return char(arch)
    name = {"win64": "PCWIN64", "maci64": "MACI64", "maca64": "MACA64", "glnxa64": "GLNXA64"}.get(arch, arch.upper())
    outs = (char(name), float(2 ** 48 - 1), char("L" if sys.byteorder == "little" else "B"))
    return outs[:nargout] if nargout > 1 else outs[0]


@builtin("getenv")
def _getenv(interp, args, nargout):
    return char(os.environ.get(text(args[0]), ""))


@builtin("help", "doc")
def _help(interp, args, nargout):
    from ..help import help_text
    if not args:
        text_ = ("Type 'help <name>' for help on a function, e.g. help plot.\n"
                 "mylab runs MATLAB-language code; see docs/FUNCTIONS.md for every function.\n")
    else:
        text_ = help_text(text(args[0]), interp)
    if nargout > 0:
        return char(text_)
    interp.out(text_)
    return None


# -- files and paths --------------------------------------------------------------------------

@builtin("pwd")
def _pwd(interp, args, nargout):
    return char(os.getcwd())


@builtin("cd")
def _cd(interp, args, nargout):
    if args:
        os.chdir(os.path.expanduser(text(args[0])))
    elif nargout == 0:
        interp.out(os.getcwd() + "\n")
        return None
    return char(os.getcwd())


@builtin("addpath")
def _addpath(interp, args, nargout):
    old = char(os.pathsep.join(interp.path))
    at_end = False
    folders: list[str] = []
    for a in args:
        if not is_text(a):
            at_end = bool(num(a).flat[0])  # addpath(..., 0) is '-begin', 1 is '-end'
            continue
        s = text(a)
        if s.lower() in ("-end", "-begin", "-frozen"):
            at_end = s.lower() == "-end" or (at_end and s.lower() == "-frozen")
            continue
        for p in s.split(os.pathsep):
            if p.strip():
                p = os.path.abspath(os.path.expanduser(p.strip()))
                if p not in folders:
                    folders.append(p)
    for p in folders:
        if p in interp.path:
            interp.path.remove(p)
    if at_end:
        interp.path.extend(folders)
    else:
        interp.path[:0] = folders
    return old if nargout else None


@builtin("rmpath")
def _rmpath(interp, args, nargout):
    for a in args:
        for p in text(a).split(os.pathsep):
            p = os.path.abspath(os.path.expanduser(p)) if p.strip() else ""
            if p in interp.path:
                interp.path.remove(p)
    return None


@builtin("path")
def _path(interp, args, nargout):
    return char(os.pathsep.join(interp.path))


@builtin("fullfile")
def _fullfile(interp, args, nargout):
    return char(os.path.join(*[text(a) for a in args]) if args else "")


@builtin("fileparts")
def _fileparts(interp, args, nargout):
    p = text(args[0])
    d, f = os.path.split(p)
    name, ext = os.path.splitext(f)
    return char(d), char(name), char(ext)


@builtin("isfile")
def _isfile(interp, args, nargout):
    return bool_scalar(os.path.isfile(text(args[0])))


@builtin("isfolder", "isdir")
def _isfolder(interp, args, nargout):
    return bool_scalar(os.path.isdir(text(args[0])))


@builtin("mkdir")
def _mkdir(interp, args, nargout):
    os.makedirs(os.path.join(*[text(a) for a in args]), exist_ok=True)
    return bool_scalar(True) if nargout else None


@builtin("delete")
def _delete(interp, args, nargout):
    from ..objects import MObject
    if args and isinstance(args[0], MObject):
        from .objfuncs import delete_object
        delete_object(interp, args[0])
        return None
    import glob
    for a in args:
        if isinstance(a, np.ndarray) and a.size == 0 and not is_text(a):
            continue  # delete([]) and delete(gcp('nocreate')) with no pool do nothing
        if isinstance(a, np.ndarray) and a.dtype.kind in "fiu" and not is_text(a):
            from ._registry import BUILTINS
            from .plotting import _delete_handle, numbered_handle
            for h in a.flatten():
                if float(h) != int(h) and numbered_handle(h) is not None:  # a line, axes, ... by number
                    _delete_handle(interp, numbered_handle(h))
                else:  # figure numbers, as close accepts them
                    BUILTINS["close"](interp, [np.array([[float(h)]])], 0)
            continue
        if isinstance(a, Struct) and "NumWorkers" in a.fields:
            continue  # the pool from parpool: there are no worker processes to shut down
        if not is_text(a):
            raise MatlabError("delete: the argument must be a file name or a handle object.")
        name = text(a)
        matches = [p for p in glob.glob(name) if os.path.isfile(p)] if any(c in name for c in "*?[") \
            else ([name] if os.path.isfile(name) else [])
        if not matches:
            interp.warn(f"File '{name}' not found.", "MATLAB:DELETE:FileNotFound")
        for path in matches:
            os.remove(path)
    return None


@builtin("fileread")
def _fileread(interp, args, nargout):
    with open(text(args[0]), encoding="utf-8") as fh:
        return char(fh.read())


class MFile:
    """An open file. Bytes on disk; text helpers decode with the handle's encoding.

    ``read``, ``readline`` and ``write`` take and return ``str`` so that the text
    functions (fprintf, fgetl, fscanf, textscan) need not know about bytes; fread
    and fwrite use ``raw`` directly.
    """

    def __init__(self, raw, name: str, perm: str, endian: str, encoding: str):
        self.raw = raw
        self.name = name
        self.perm = perm
        self.endian = endian  # '<' or '>'
        self.encoding = encoding
        self.hit_eof = False
        self.wbits = [0, 0]  # bits written with a bit precision that do not fill a byte yet: value, count
        self.rbits = [0, 0]  # bits read from the file but not yet handed out
        self.bit_order = endian

    def sync(self) -> None:
        """Finish bit-field I/O before anything else touches the file: pad the last byte with zeros."""
        if self.wbits[1]:
            value, count = self.wbits
            self.wbits = [0, 0]
            byte = value & 0xFF if self.bit_order == "<" else (value << (8 - count)) & 0xFF
            self.raw.write(bytes([byte]))
        self.rbits = [0, 0]

    def write_bits(self, values: list[int], width: int, order: str) -> None:
        if self.rbits[1] or (self.wbits[1] and order != self.bit_order):
            self.sync()
        self._switch()
        self.bit_order = order
        acc, count = self.wbits
        out = bytearray()
        mask = (1 << width) - 1
        for v in values:
            v &= mask
            if order == "<":  # the first value sits in the low bits of the first byte
                acc |= v << count
                count += width
                while count >= 8:
                    out.append(acc & 0xFF)
                    acc >>= 8
                    count -= 8
            else:
                acc = (acc << width) | v
                count += width
                while count >= 8:
                    out.append((acc >> (count - 8)) & 0xFF)
                    count -= 8
                    acc &= (1 << count) - 1
        self.wbits = [acc, count]
        if out:
            self.raw.write(bytes(out))

    def read_bits(self, width: int, order: str, skip: int = 0) -> int | None:
        """The next ``width`` bits as an unsigned number, or None at the end of the file."""
        if self.wbits[1] or (self.rbits[1] and order != self.bit_order):
            self.sync()
        self.bit_order = order
        acc, count = self.rbits
        need = width + skip
        while count < need:
            b = self.raw.read(1)
            if not b:
                self.rbits = [acc, count]
                return None
            if order == "<":
                acc |= b[0] << count
            else:
                acc = (acc << 8) | b[0]
            count += 8
        if order == "<":
            value = acc & ((1 << width) - 1)
            acc >>= need
        else:
            value = (acc >> (count - width)) & ((1 << width) - 1)
            acc &= (1 << (count - need)) - 1
        self.rbits = [acc, count - need]
        return value

    def decode(self, data: bytes) -> str:
        try:
            return data.decode(self.encoding)
        except (UnicodeDecodeError, LookupError):
            return data.decode("latin-1")

    def encode(self, s: str) -> bytes:
        try:
            return s.encode(self.encoding)
        except (UnicodeEncodeError, LookupError):
            return s.encode("utf-8")

    def read(self, n: int = -1) -> str:
        self.sync()
        if n is None or n < 0:
            return self.decode(self.raw.read())
        out = self.raw.read(n)
        if self.encoding.lower().replace("-", "") == "utf8":
            # finish a multi-byte character that was cut in half
            while out and (out[-1] & 0x80):
                try:
                    out.decode("utf-8")
                    break
                except UnicodeDecodeError as e:
                    if e.reason != "unexpected end of data":
                        break
                    more = self.raw.read(1)
                    if not more:
                        break
                    out += more
        return self.decode(out)

    def readline(self) -> str:
        self.sync()
        return self.decode(self.raw.readline())

    def write(self, s: str) -> int:
        self.sync()
        self._switch()
        data = self.encode(s)
        self.raw.write(data)
        return len(data)

    def write_bytes(self, data: bytes) -> None:
        self.sync()
        self._switch()
        self.raw.write(data)

    def _switch(self) -> None:
        # buffered streams need a seek between reading and writing
        if "+" in self.perm and self.raw.seekable():
            self.raw.seek(self.raw.tell())

    def tell(self) -> int:
        return self.raw.tell()

    def seek(self, pos: int, whence: int = 0) -> int:
        self.sync()
        return self.raw.seek(pos, whence)

    def size(self) -> int:
        if self.wbits[1]:
            self.sync()
        pos = self.raw.tell()
        end = self.raw.seek(0, 2)
        self.raw.seek(pos)
        return end

    def at_end(self) -> bool:
        if "r" not in self.perm and "+" not in self.perm:
            return False
        pos = self.raw.tell()
        end = self.raw.read(1) == b""
        self.raw.seek(pos)
        return end

    def flush(self) -> None:
        if self.wbits[1]:
            self.sync()
        self.raw.flush()

    def close(self) -> None:
        if self.wbits[1]:
            self.sync()
        self.raw.close()


_NATIVE = "<" if sys.byteorder == "little" else ">"
_MACHINE_FORMATS = {
    "n": _NATIVE, "native": _NATIVE,
    "l": "<", "ieee-le": "<", "a": "<", "ieee-le.l64": "<",
    "b": ">", "ieee-be": ">", "s": ">", "ieee-be.l64": ">",
}


def machine_format(v, default: str = "<") -> str:
    """Byte-order character ('<' or '>') for a machine-format name such as 'ieee-be'."""
    s = text(v, "machine format").lower()
    if s == "":
        return default
    if s not in _MACHINE_FORMATS:
        raise MatlabError(f"Unsupported machine format '{s}'. Use 'n', 'l' ('ieee-le') or 'b' ('ieee-be').")
    return _MACHINE_FORMATS[s]


def file_handle(interp, fid: int):
    files = interp.state.setdefault("files", {})
    if fid not in files:
        raise MatlabError("Invalid file identifier. Use fopen to generate a valid file identifier.")
    return files[fid]


def _fid(v) -> int:
    if is_text(v):
        raise MatlabError("Invalid file identifier. Use fopen to generate a valid file identifier.")
    return to_int(v, "file identifier")


@builtin("fopen")
def _fopen(interp, args, nargout):
    nargs(args, 1, 4, "fopen")
    files = interp.state.setdefault("files", {})
    if not is_text(args[0]):
        # fopen(fid): describe an open file
        fid = to_int(args[0], "file identifier")
        if fid in (0, 1, 2):
            outs = (char(("stdin", "stdout", "stderr")[fid]), char("r" if fid == 0 else "a"),
                    char("ieee-le"), char("UTF-8"))
        elif fid in files:
            f = files[fid]
            outs = (char(f.name), char(f.perm), char("ieee-le" if f.endian == "<" else "ieee-be"),
                    char(f.encoding))
        else:
            outs = (char(""), char(""), char(""), char(""))
        return outs[:nargout] if nargout > 1 else outs[0]
    name = text(args[0])
    if len(args) == 1 and name == "all":
        return np.array(sorted(files), dtype=float).reshape(-1, 1) if files else np.zeros((0, 0))
    perm = text(args[1]) if len(args) > 1 else "r"
    mode = perm.lower().replace("t", "").replace("b", "")
    if mode not in ("r", "w", "a", "r+", "w+", "a+"):
        raise MatlabError(f"Invalid permission '{perm}'.")
    endian = machine_format(args[2], _NATIVE) if len(args) > 2 else _NATIVE
    encoding = (text(args[3]) if len(args) > 3 else "") or "UTF-8"  # '' means the default encoding
    try:
        "".encode(encoding)
    except LookupError:
        raise MatlabError(f"Unsupported encoding '{encoding}'.")
    name = os.path.expanduser(name)
    path = name
    if mode in ("r", "r+") and not os.path.exists(path) and not os.path.isabs(path):
        # data files are also looked up on the search path
        path = next((c for c in (os.path.join(d, name) for d in interp.path) if os.path.isfile(c)), name)
    try:
        raw = open(path, mode + "b")
    except OSError as e:
        if nargout > 1:
            return -1.0, char(e.strerror or "Cannot open file")
        return -1.0
    fid = 3
    while fid in files:
        fid += 1
    files[fid] = MFile(raw, os.path.abspath(path), perm, endian, encoding)
    return (float(fid), char("")) if nargout > 1 else float(fid)


@builtin("fclose")
def _fclose(interp, args, nargout):
    nargs(args, 1, 1, "fclose")
    files = interp.state.setdefault("files", {})
    if is_text(args[0]):
        if text(args[0]) != "all":
            raise MatlabError("Invalid file identifier. Use fopen to generate a valid file identifier.")
        for fh in files.values():
            fh.close()
        files.clear()
        return 0.0 if nargout else None
    for fid in [int(v) for v in num(args[0]).flatten()]:
        if fid in (0, 1, 2):
            continue
        fh = files.pop(fid, None)
        if fh is None:
            raise MatlabError("Invalid file identifier.")
        fh.close()
    return 0.0 if nargout else None


def _readline(fh) -> str:
    line = fh.readline()
    if line == "":
        fh.hit_eof = True
    return line


@builtin("fgetl")
def _fgetl(interp, args, nargout):
    nargs(args, 1, 1, "fgetl")
    line = _readline(file_handle(interp, _fid(args[0])))
    if line == "":
        return -1.0
    if line.endswith("\n"):
        line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
    return char(line)


@builtin("fgets")
def _fgets(interp, args, nargout):
    nargs(args, 1, 2, "fgets")
    fh = file_handle(interp, _fid(args[0]))
    if len(args) > 1:
        # at most nchar characters of the line
        limit = to_int(args[1], "nchar")
        start = fh.tell()
        line = _readline(fh)
        if len(line) > limit:
            line = line[:limit]
            fh.seek(start + len(fh.encode(line)))
    else:
        line = _readline(fh)
    return -1.0 if line == "" else char(line)


@builtin("feof")
def _feof(interp, args, nargout):
    nargs(args, 1, 1, "feof")
    fid = _fid(args[0])
    if fid in (0, 1, 2):
        return bool_scalar(False)
    return bool_scalar(file_handle(interp, fid).at_end())


# -- binary files ------------------------------------------------------------------------------

_PRECISIONS = {
    "uint8": "u1", "uchar": "u1", "unsignedchar": "u1", "char": "u1", "char*1": "u1", "integer*1": "i1",
    "int8": "i1", "schar": "i1", "signedchar": "i1",
    "int16": "i2", "short": "i2", "integer*2": "i2", "uint16": "u2", "ushort": "u2", "unsignedshort": "u2",
    "int32": "i4", "int": "i4", "long": "i4", "integer*4": "i4",
    "uint32": "u4", "uint": "u4", "ulong": "u4", "unsignedint": "u4", "unsignedlong": "u4",
    "int64": "i8", "integer*8": "i8", "uint64": "u8",
    "single": "f4", "float32": "f4", "float": "f4", "real*4": "f4",
    "double": "f8", "float64": "f8", "real*8": "f8",
}
_CLASS_OF = {"u1": "uint8", "i1": "int8", "u2": "uint16", "i2": "int16", "u4": "uint32", "i4": "int32",
             "u8": "uint64", "i8": "int64", "f4": "single", "f8": "double"}
_CHAR_NAMES = {"char", "uchar", "schar", "char*1", "unsignedchar", "signedchar"}


def _precision_code(name: str) -> str:
    """NumPy type code for a precision name; bit fields give 'ubitN' or 'bitN' (1 to 64 bits)."""
    key = name.lower().replace(" ", "")
    if key not in _PRECISIONS:
        m = re.fullmatch(r"(u?)bit(\d+)", key)
        if m:
            if not 1 <= int(m.group(2)) <= 64:
                raise MatlabError(f"The bit-field precision '{name}' must have between 1 and 64 bits.")
            return key
        raise MatlabError(f"Unknown precision '{name}'.")
    return _PRECISIONS[key]


def _bit_width(code: str) -> tuple[int, bool] | None:
    """(number of bits, signed) for a bit-field code, None for an ordinary type code."""
    if "bit" not in code:
        return None
    return int(code.split("bit")[1]), not code.startswith("u")


def parse_precision(spec: str) -> tuple[str, str, int]:
    """(numpy type code of the file data, output class, block length) for a precision string."""
    s = spec.strip()
    block = 1
    m = re.match(r"^(\d+)\s*\*\s*(.+)$", s)
    if m:
        block = max(int(m.group(1)), 1)
        s = m.group(2).strip()
    same = s.startswith("*")
    if same:
        s = s[1:].strip()
    if "=>" in s:
        src, dst = (t.strip() for t in s.split("=>", 1))
    else:
        src, dst = s, None
    code = _precision_code(src)
    bits = _bit_width(code)
    if dst is None:
        if same and bits is not None:
            # the smallest integer class that holds the field
            size = next(n for n in (8, 16, 32, 64) if n >= bits[0])
            out = f"{'int' if bits[1] else 'uint'}{size}"
        elif same:
            out = "char" if src.lower().replace(" ", "") in ("char", "char*1") else _CLASS_OF[code]
        else:
            out = "double"
    else:
        low = dst.lower().replace(" ", "")
        if low in _CHAR_NAMES:
            out = "char"
        elif low == "logical":
            out = "logical"
        else:
            dcode = _precision_code(dst)
            if _bit_width(dcode) is not None:
                raise MatlabError(f"The output class in '{spec}' must be a class name such as 'double' or 'uint8'.")
            out = _CLASS_OF[dcode]
    return code, out, block


def _read_values(fh, dtype, want, block: int, skip: int) -> np.ndarray:
    item = dtype.itemsize
    if skip == 0:
        data = fh.raw.read() if want is None else fh.raw.read(want * item)
        usable = len(data) - len(data) % item
        return np.frombuffer(data[:usable], dtype=dtype)
    # blocks of `block` values, each followed by `skip` bytes that are ignored
    chunks = []
    got = 0
    total = fh.size()
    while want is None or got < want:
        n = block if want is None else min(block, want - got)
        data = fh.raw.read(n * item)
        usable = len(data) - len(data) % item
        if usable:
            chunks.append(np.frombuffer(data[:usable], dtype=dtype))
            got += usable // item
        if usable < n * item:
            break
        target = fh.raw.tell() + skip  # the skip also follows the last block read
        if target > total or target < 0:
            break
        fh.raw.seek(target)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)


def _read_bit_fields(fh, bits: tuple[int, bool], want, block: int, skip: int, order: str) -> np.ndarray:
    """Values of a bit-field precision; ``skip`` bits are passed over after every ``block`` values."""
    width, signed = bits
    out: list[int] = []
    while want is None or len(out) < want:
        v = fh.read_bits(width, order)
        if v is None:
            break
        if signed and v >= 1 << (width - 1):
            v -= 1 << width
        out.append(v)
        if skip and len(out) % block == 0:
            if fh.read_bits(0, order, skip) is None:
                break
    return np.array(out, dtype=np.int64 if signed else np.uint64)


@builtin("fread")
def _fread(interp, args, nargout):
    nargs(args, 1, 5, "fread")
    fh = file_handle(interp, _fid(args[0]))
    rest = list(args[1:])
    size = None
    if rest and not is_text(rest[0]):
        size = num(rest.pop(0)).flatten().astype(float)
    spec = text(rest.pop(0)) if rest and is_text(rest[0]) else "uint8"
    skip = 0
    if rest and not is_text(rest[0]):
        skip = to_int(rest.pop(0), "skip")
    endian = machine_format(rest.pop(0), fh.endian) if rest else fh.endian
    code, out_class, block = parse_precision(spec)
    bits = _bit_width(code)
    dtype = np.dtype(endian + code) if bits is None else np.dtype(np.int64 if bits[1] else np.uint64)
    shape = None
    if size is None or size.size == 0:
        want = None
    elif size.size == 1:
        want = None if np.isinf(size[0]) else int(size[0])
    elif size.size == 2:
        rows = int(size[0])
        want = None if np.isinf(size[1]) else rows * int(size[1])
        shape = (rows, None if np.isinf(size[1]) else int(size[1]))
    else:
        raise MatlabError("fread: size must be n, Inf or [m n].")
    if want is not None and want < 0:
        raise MatlabError("fread: size must not be negative.")
    if bits is None:
        fh.sync()
        values = _read_values(fh, dtype, want, block, skip)
    else:
        values = _read_bit_fields(fh, bits, want, block, skip, endian)
    count = values.size
    if want is None or count < want:
        fh.hit_eof = True
    if out_class == "char":
        if code == "u1" and values.size and values.max() > 127 and \
                fh.encoding.lower().replace("-", "") == "utf8":
            out = np.array(list(fh.decode(values.tobytes())), dtype="<U1")
        else:
            out = np.array([chr(int(v)) if v >= 0 else "\0" for v in values], dtype="<U1")
    elif out_class == "logical":
        out = values != 0
    elif out_class == "double":
        out = values.astype(float)
    elif out_class == "single":
        out = values.astype(np.float32)
    else:
        from ..values import INT_TYPES, saturate_int
        target = INT_TYPES[out_class]
        if values.dtype.kind in "iu":
            info = np.iinfo(target)
            out = np.clip(values.astype(object), info.min, info.max).astype(target) if values.size else \
                np.zeros(0, dtype=target)
        else:
            out = saturate_int(values.astype(float), target)
    n = out.size
    if n == 0:
        result = np.zeros((0, 0), dtype=out.dtype)
    elif shape is None:
        result = out.reshape(-1, 1)
    else:
        rows = shape[0]
        cols = shape[1] if shape[1] is not None else -(-n // rows)
        if n < rows * cols or shape[1] is None:
            # a short read keeps whole columns, zero-filling the last one
            cols = -(-n // rows)
            pad = np.zeros(rows * cols, dtype=out.dtype)
            if out.dtype.kind == "U":
                pad[:] = "\0"
            pad[:n] = out
            out = pad
        result = out.reshape((rows, cols), order="F")
    return (result, float(count)) if nargout > 1 else result


@builtin("fwrite")
def _fwrite(interp, args, nargout):
    nargs(args, 2, 5, "fwrite")
    fid = _fid(args[0])
    a = args[1]
    rest = list(args[2:])
    spec = text(rest.pop(0)) if rest and is_text(rest[0]) else "uint8"
    skip = 0
    if rest and not is_text(rest[0]):
        skip = to_int(rest.pop(0), "skip")
    was_text = is_text(a)
    if isinstance(a, MString):
        a = char(str(a))
    if not isinstance(a, np.ndarray):
        raise MatlabError("fwrite: the data must be a numeric, logical or char array.")
    flat = a.flatten(order="F")
    count = flat.size
    if fid in (1, 2):
        interp.out("".join(chr(int(v)) for v in np.real(num(flat))))
        return float(count) if nargout else None
    fh = file_handle(interp, fid)
    endian = machine_format(rest.pop(0), fh.endian) if rest else fh.endian
    code, _out, _block = parse_precision(spec)
    bits = _bit_width(code)
    if bits is not None:
        width, signed = bits
        lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
        vals = np.real(num(flat)).astype(float)
        vals = np.where(np.isnan(vals), 0, np.round(vals))
        ints = [int(min(max(v, lo), hi)) for v in vals]
        for v in ints:
            if skip:
                fh.write_bits([0] * skip, 1, endian)  # skipped bits are written as zeros
            fh.write_bits([v], width, endian)
        return float(count) if nargout else None
    dtype = np.dtype(endian + code)
    if was_text and spec.lower().replace(" ", "").lstrip("*") in ("char", "char*1") and skip == 0:
        fh.write("".join(flat) if count else "")
        return float(count) if nargout else None
    vals = np.real(num(flat))
    if dtype.kind in "iu":
        if vals.dtype.kind in "iu":
            info = np.iinfo(dtype)
            data = np.clip(vals.astype(object), info.min, info.max).astype(dtype) if count else \
                np.zeros(0, dtype=dtype)
        else:
            from ..values import saturate_int
            data = saturate_int(vals, np.dtype(code).type).astype(dtype)
    else:
        data = vals.astype(dtype)
    if skip == 0:
        fh.write_bytes(data.tobytes())
    else:
        fh._switch()
        size = data.dtype.itemsize
        raw = data.tobytes()
        for k in range(count):
            # skip bytes before each value, keeping whatever the file already holds there
            pos = fh.raw.tell()
            end = fh.size()
            if pos + skip > end:
                fh.raw.seek(0, 2)
                fh.raw.write(b"\0" * (pos + skip - end))
            else:
                fh.raw.seek(pos + skip)
            fh.raw.write(raw[k * size:(k + 1) * size])
    return float(count) if nargout else None


@builtin("fseek")
def _fseek(interp, args, nargout):
    nargs(args, 2, 3, "fseek")
    fh = file_handle(interp, _fid(args[0]))
    offset = to_int(args[1], "offset")
    origin = args[2] if len(args) > 2 else char("bof")
    bad = "fseek: origin must be 'bof', 'cof' or 'eof' (or -1, 0 or 1)."
    if is_text(origin):
        key = text(origin).lower()
        if key not in ("bof", "cof", "eof"):
            raise MatlabError(bad)
        whence = {"bof": 0, "cof": 1, "eof": 2}[key]
    else:
        code = to_int(origin, "origin")
        if code not in (-1, 0, 1):
            raise MatlabError(bad)
        whence = code + 1
    fh.flush()
    target = (0, fh.tell(), fh.size())[whence] + offset
    ok = 0 <= target <= fh.size()
    if ok:
        fh.seek(target)
        fh.hit_eof = False
        interp.state["ferror"] = ""
    else:
        interp.state["ferror"] = "Offset is bad - after end-of-file or before beginning-of-file."
    return (0.0 if ok else -1.0) if nargout else None


@builtin("ftell")
def _ftell(interp, args, nargout):
    nargs(args, 1, 1, "ftell")
    return float(file_handle(interp, _fid(args[0])).tell())


@builtin("frewind")
def _frewind(interp, args, nargout):
    nargs(args, 1, 1, "frewind")
    fh = file_handle(interp, _fid(args[0]))
    fh.seek(0)
    fh.hit_eof = False
    return None


@builtin("ferror")
def _ferror(interp, args, nargout):
    nargs(args, 1, 2, "ferror")
    fid = _fid(args[0])
    if fid not in (0, 1, 2):
        file_handle(interp, fid)
    msg = interp.state.get("ferror", "")
    if len(args) > 1 and text(args[1]).lower() == "clear":
        interp.state["ferror"] = ""
    return (char(msg), -1.0 if msg else 0.0) if nargout > 1 else char(msg)


@builtin("fflush")
def _fflush(interp, args, nargout):
    if args and not is_text(args[0]):
        fh = interp.state.setdefault("files", {}).get(to_int(args[0], "file identifier"))
        if fh is not None:
            fh.flush()
    return None


# save and load live in fileformats.py (MAT-files) with the rest of the data file formats.


@builtin("run")
def _run(interp, args, nargout):
    """Run a script file in the caller's workspace."""
    from ..parser import parse
    path = text(args[0])
    if path.lower().endswith(".mlapp"):
        from ..ui import MLAPP_MESSAGE
        raise MatlabError(MLAPP_MESSAGE)
    if not path.endswith(".m") and os.path.isfile(path + ".m"):
        path += ".m"
    if not os.path.isfile(path):
        found = interp.find_file_function(os.path.splitext(os.path.basename(path))[0])
        if found is None:
            raise MatlabError(f"run: file '{path}' not found.")
        path = found[1]
    path = os.path.abspath(path)
    from ..sourcefile import read_source
    program = parse(read_source(path))
    directory = os.path.dirname(path)
    if directory not in interp.path:
        interp.path.insert(0, directory)
    if not program.body and program.functions:
        main = next(iter(program.functions.values()))
        interp.call_user_function(main, [], 0, program.functions, path)
        return None
    interp.run_program(program, interp.frame, path)
    return None


@builtin("import")
def _import(interp, args, nargout):
    """``import pkg.*`` / ``import pkg.name``: later calls in this workspace may leave out the package."""
    frame = interp.frame
    imports = frame.imports
    for a in args:
        name = text(a)
        if not name:
            continue
        if imports is None:
            imports = frame.imports = []  # one list per workspace: a function's imports end with it
        if name not in imports:
            imports.append(name)
    if nargout:
        return Cell.from_list([char(n) for n in imports or []], column=True)
    return None
