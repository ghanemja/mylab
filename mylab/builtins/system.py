"""Output, errors, workspace management, evaluation, files and timing."""

from __future__ import annotations

import os
import re
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
        frame.globals.pop(pattern, None)
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
    mode = "".join(text(a).lower() for a in args) if args else "short"
    mode = mode.replace(" ", "")
    if mode in ("short", "long", "shortg", "longg", "shorte", "longe"):
        FORMAT.mode = mode
    elif mode in ("compact", "loose", "default"):
        if mode == "default":
            FORMAT.mode = "short"
    else:
        raise MatlabError(f"Unsupported format '{mode}'. Use short, long, short g, long g, short e or long e.")
    return None


@builtin("more", "beep", "commandwindow", "pack", "dbstop", "profile")
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
    for a in args:
        if is_text(a) and os.path.isfile(text(a)):
            os.remove(text(a))
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
        return self.decode(self.raw.readline())

    def write(self, s: str) -> int:
        self._switch()
        data = self.encode(s)
        self.raw.write(data)
        return len(data)

    def write_bytes(self, data: bytes) -> None:
        self._switch()
        self.raw.write(data)

    def _switch(self) -> None:
        # buffered streams need a seek between reading and writing
        if "+" in self.perm and self.raw.seekable():
            self.raw.seek(self.raw.tell())

    def tell(self) -> int:
        return self.raw.tell()

    def seek(self, pos: int, whence: int = 0) -> int:
        return self.raw.seek(pos, whence)

    def size(self) -> int:
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
        self.raw.flush()

    def close(self) -> None:
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
    encoding = text(args[3]) if len(args) > 3 else "UTF-8"
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
    key = name.lower().replace(" ", "")
    if key not in _PRECISIONS:
        if re.fullmatch(r"u?bit\d+", key):
            raise MatlabError(f"Bit-field precisions such as '{name}' are not supported.")
        raise MatlabError(f"Unknown precision '{name}'.")
    return _PRECISIONS[key]


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
    if dst is None:
        if same:
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
            out = _CLASS_OF[_precision_code(dst)]
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
    dtype = np.dtype(endian + code)
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
    values = _read_values(fh, dtype, want, block, skip)
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


@builtin("import")
def _import(interp, args, nargout):
    """Accepts package imports such as ``import pkg.*``; mylab has no packages, so this does nothing."""
    if nargout:
        return Cell(np.empty((1, 0), dtype=object))
    return None
