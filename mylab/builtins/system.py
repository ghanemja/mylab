"""Output, errors, workspace management, evaluation, files and timing."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from .. import __version__
from ..display import FORMAT, display_lines, format_display, function_handle_text
from ..errors import MatlabError
from ..values import (
    Cell, FunctionHandle, MString, Struct, char, class_name, is_char, shape_of, to_int, to_str,
)
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, is_text, nargs, num, text
from .strings import matlab_sprintf


@builtin("disp")
def _disp(interp, args, nargout):
    nargs(args, 1, 1, "disp")
    lines = display_lines(args[0])
    if lines:
        interp.out("\n".join(lines) + "\n")
    elif isinstance(args[0], (np.ndarray,)) and args[0].size == 0 and is_char(args[0]):
        interp.out("\n")
    return None


@builtin("display")
def _display(interp, args, nargout):
    interp.out(format_display("ans", args[0]))
    return None


def _error_from_args(args) -> MatlabError:
    from ..objects import MExceptionObject
    if len(args) == 1 and isinstance(args[0], MExceptionObject):
        return args[0].to_error()
    if len(args) == 1 and isinstance(args[0], Struct):
        s = args[0].data.flat[0]
        return MatlabError(to_str(s.get("message", char(""))), to_str(s.get("identifier", char(""))))
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


@builtin("warning")
def _warning(interp, args, nargout):
    if not args:
        return None
    first = text(args[0])
    if first.lower() in ("off", "on") and len(args) <= 2:
        enable = first.lower() == "on"
        if len(args) == 1 or text(args[1]).lower() == "all":
            interp.warnings_enabled = enable
            if enable:
                interp.disabled_warnings.clear()
        elif enable:
            interp.disabled_warnings.discard(text(args[1]))
        else:
            interp.disabled_warnings.add(text(args[1]))
        return None
    if first.lower() == "query":
        ident = text(args[1]) if len(args) > 1 else "all"
        on = interp.warnings_enabled and ident not in interp.disabled_warnings
        return Struct.scalar({"identifier": char(ident), "state": char("on" if on else "off")}) if nargout else None
    rest = list(args[1:])
    ident = ""
    if rest and ":" in first and " " not in first:
        ident = first
        first = text(rest[0])
        rest = rest[1:]
    interp.warn(matlab_sprintf(first, rest) if (rest or ident) else first, ident)
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


@builtin("pause")
def _pause(interp, args, nargout):
    if args and not is_text(args[0]):
        secs = float(num(args[0]).flat[0])
        if np.isfinite(secs) and secs > 0:
            time.sleep(min(secs, 3600))
    return None


@builtin("clock")
def _clock(interp, args, nargout):
    t = time.localtime()
    return np.array([[t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min,
                      t.tm_sec + (time.time() % 1)]], dtype=float)


@builtin("cputime")
def _cputime(interp, args, nargout):
    return time.process_time()


@builtin("now")
def _now(interp, args, nargout):
    # days since year 0 (MATLAB datenum convention)
    return time.time() / 86400.0 + 719529.0


@builtin("datestr")
def _datestr(interp, args, nargout):
    t = time.time() if not args else (float(num(args[0]).flat[0]) - 719529.0) * 86400.0
    return char(time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(t)))


# -- workspace -------------------------------------------------------------------------

@builtin("clc", "home")
def _clc(interp, args, nargout):
    hook = interp.state.get("clc_hook")
    if hook:
        hook()
    return None


@builtin("clear", "clearvars")
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
            interp.global_vars.clear()
            from .plotting import close_all
            close_all(interp)
        return None
    if names[0] in ("global", "-global"):
        for n in names[1:] or list(interp.global_vars):
            interp.global_vars.pop(n, None)
        return None
    if names[0] in ("functions",):
        return None
    import fnmatch
    for pattern in names:
        for n in list(frame.vars):
            if fnmatch.fnmatchcase(n, pattern):
                del frame.vars[n]
        frame.globals.discard(pattern)
    return None


@builtin("who")
def _who(interp, args, nargout):
    names = sorted(interp.frame.vars)
    if nargout > 0:
        return Cell.from_list([char(n) for n in names], column=True)
    if names:
        interp.out("\nYour variables are:\n\n" + "  ".join(names) + "\n\n")
    return None


@builtin("whos")
def _whos(interp, args, nargout):
    frame = interp.frame
    rows = []
    for n in sorted(frame.vars):
        v = frame.vars[n]
        r, c = shape_of(v)
        nbytes = v.nbytes if isinstance(v, np.ndarray) else 0
        rows.append((n, f"{r}x{c}", nbytes, class_name(v)))
    if not rows:
        return None
    w = max(4, max(len(r[0]) for r in rows))
    lines = [f"  {'Name':<{w}}      Size            Bytes  Class", ""]
    for n, size, nbytes, cls in rows:
        lines.append(f"  {n:<{w}}      {size:<12}{nbytes:>9}  {cls}")
    interp.out("\n".join(lines) + "\n\n")
    return None


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
            return 2.0
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


@builtin("nargin")
def _nargin(interp, args, nargout):
    if args:
        raise MatlabError("nargin(fname) is not supported yet.")
    return float(interp.frame.nargin)


@builtin("nargout")
def _nargout(interp, args, nargout):
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
    try:
        interp.run_program(parse(text(args[0])), interp.frame)
    finally:
        interp.out = old
    return char("".join(captured))


@builtin("input")
def _input(interp, args, nargout):
    prompt = text(args[0]) if args else ""
    reply = interp.input_func(prompt)
    if len(args) > 1 and text(args[1]) == "s":
        return char(reply)
    if not reply.strip():
        return np.zeros((0, 0))
    return interp.eval_expression(reply)


@builtin("keyboard")
def _keyboard(interp, args, nargout):
    return None


@builtin("format")
def _format(interp, args, nargout):
    mode = text(args[0]).lower() if args else "short"
    if mode in ("long", "longg", "long g"):
        FORMAT.long = True
    elif mode in ("short", "shortg", "short g", "compact", "loose"):
        if mode not in ("compact", "loose"):
            FORMAT.long = False
    return None


@builtin("more", "beep", "commandwindow", "pack", "dbstop", "profile")
def _noop(interp, args, nargout):
    return None


@builtin("version")
def _version(interp, args, nargout):
    return char(f"mylab {__version__}")


@builtin("ver")
def _ver(interp, args, nargout):
    interp.out(f"mylab version {__version__} (Python {sys.version.split()[0]}, NumPy {np.__version__})\n")
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


@builtin("computer")
def _computer(interp, args, nargout):
    return char(sys.platform)


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
    for a in args:
        s = text(a)
        if s.startswith("-"):
            continue
        for p in s.split(os.pathsep):
            p = os.path.abspath(os.path.expanduser(p))
            if p not in interp.path:
                interp.path.insert(0, p)
    return None


@builtin("rmpath")
def _rmpath(interp, args, nargout):
    for a in args:
        p = os.path.abspath(text(a))
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
    for a in args:
        if is_text(a) and os.path.isfile(text(a)):
            os.remove(text(a))
    return None


@builtin("fileread")
def _fileread(interp, args, nargout):
    with open(text(args[0]), encoding="utf-8") as fh:
        return char(fh.read())


def file_handle(interp, fid: int):
    files = interp.state.setdefault("files", {})
    if fid not in files:
        raise MatlabError("Invalid file identifier. Use fopen to generate a valid file identifier.")
    return files[fid]


@builtin("fopen")
def _fopen(interp, args, nargout):
    name = text(args[0])
    perm = text(args[1]) if len(args) > 1 else "r"
    mode = perm.replace("t", "").replace("b", "")
    if mode not in ("r", "w", "a", "r+", "w+", "a+"):
        raise MatlabError(f"Invalid permission '{perm}'.")
    try:
        fh = open(name, mode, encoding="utf-8")
    except OSError as e:
        if nargout > 1:
            return -1.0, char(str(e))
        return -1.0
    files = interp.state.setdefault("files", {})
    fid = 3
    while fid in files:
        fid += 1
    files[fid] = fh
    return (float(fid), char("")) if nargout > 1 else float(fid)


@builtin("fclose")
def _fclose(interp, args, nargout):
    files = interp.state.setdefault("files", {})
    if args and is_text(args[0]) and text(args[0]) == "all":
        for fh in files.values():
            fh.close()
        files.clear()
        return 0.0 if nargout else None
    fid = to_int(args[0])
    fh = files.pop(fid, None)
    if fh is None:
        raise MatlabError("Invalid file identifier.")
    fh.close()
    return 0.0 if nargout else None


@builtin("fgetl")
def _fgetl(interp, args, nargout):
    line = file_handle(interp, to_int(args[0])).readline()
    if line == "":
        return -1.0
    return char(line.rstrip("\n").rstrip("\r"))


@builtin("fgets")
def _fgets(interp, args, nargout):
    line = file_handle(interp, to_int(args[0])).readline()
    return -1.0 if line == "" else char(line)


@builtin("feof")
def _feof(interp, args, nargout):
    fh = file_handle(interp, to_int(args[0]))
    pos = fh.tell()
    at_end = fh.read(1) == ""
    fh.seek(pos)
    return bool_scalar(at_end)


@builtin("writematrix", "dlmwrite", "csvwrite")
def _writematrix(interp, args, nargout):
    # writematrix(A, file) / csvwrite(file, A) / dlmwrite(file, A)
    if is_text(args[0]):
        fname, a = text(args[0]), args[1]
    else:
        a, fname = args[0], text(args[1])
    np.savetxt(fname, num(a), delimiter=",", fmt="%.15g")
    return None


@builtin("readmatrix", "csvread", "dlmread")
def _readmatrix(interp, args, nargout):
    fname = text(args[0])
    data = np.genfromtxt(fname, delimiter=",", dtype=float)
    return np.atleast_2d(data)


@builtin("save")
def _save(interp, args, nargout):
    fname = text(args[0]) if args else "matlab.mat"
    names = [text(a) for a in args[1:] if not text(a).startswith("-")]
    frame = interp.frame
    from scipy.io import savemat
    if not fname.endswith(".mat"):
        fname += ".mat"
    data = {}
    for n in names or list(frame.vars):
        v = frame.vars.get(n)
        if isinstance(v, np.ndarray):
            data[n] = v if not is_char(v) else to_str(v)
        elif isinstance(v, MString):
            data[n] = str(v)
    savemat(fname, data)
    return None


@builtin("load")
def _load(interp, args, nargout):
    fname = text(args[0])
    from scipy.io import loadmat
    if fname.endswith((".csv", ".txt", ".dat")):
        data = np.atleast_2d(np.loadtxt(fname, delimiter="," if fname.endswith(".csv") else None))
        if nargout:
            return data
        interp.frame.set(os.path.splitext(os.path.basename(fname))[0], data)
        return None
    raw = loadmat(fname if fname.endswith(".mat") else fname + ".mat")
    loaded = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind == "U":
            loaded[k] = char(str(v[0]) if v.size else "")
        elif isinstance(v, np.ndarray) and v.dtype.kind in "biufc":
            loaded[k] = np.atleast_2d(v).astype(float) if v.dtype.kind != "c" else np.atleast_2d(v)
    if nargout:
        return Struct.scalar(loaded)
    for k, v in loaded.items():
        interp.frame.set(k, v)
    return None


@builtin("run")
def _run(interp, args, nargout):
    """Run a script file in the caller's workspace."""
    from ..parser import parse
    path = text(args[0])
    if not path.endswith(".m") and os.path.isfile(path + ".m"):
        path += ".m"
    if not os.path.isfile(path):
        found = interp.find_file_function(os.path.splitext(os.path.basename(path))[0])
        if found is None:
            raise MatlabError(f"run: file '{path}' not found.")
        path = found[1]
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as fh:
        program = parse(fh.read())
    directory = os.path.dirname(path)
    if directory not in interp.path:
        interp.path.insert(0, directory)
    if not program.body and program.functions:
        main = next(iter(program.functions.values()))
        interp.call_user_function(main, [], 0, program.functions, path)
        return None
    interp.run_program(program, interp.frame, path)
    return None
