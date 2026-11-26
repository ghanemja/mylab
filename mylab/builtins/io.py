"""Text-file input and output: delimited numeric and cell data, lines, scanning, folders.

Some names here (writematrix, dlmwrite, csvwrite, readmatrix, csvread, dlmread) replace the
simpler versions in system.py; this module is imported after it, so these registrations win.
"""

from __future__ import annotations

import csv
import datetime as _dt
import glob
import os
import re
import shutil

import numpy as np

from ..errors import MatlabError
from ..values import (
    Cell, MString, StringArray, Struct, char, is_char, string_result, to_int, to_str,
)
from . import _fastcsv
from ._registry import builtin
from ._util import bool_scalar, is_text, nargs, num, text
from .system import file_handle

_DELIM_NAMES = {"tab": "\t", "\\t": "\t", "space": " ", "comma": ",", "semi": ";", "bar": "|"}


def _delimiter(v) -> str:
    d = text(v)
    return _DELIM_NAMES.get(d.lower(), d) if len(d) != 1 or d in _DELIM_NAMES else d


def _name_value(args: list, names: set[str]) -> tuple[list, dict]:
    """Split case-insensitive name-value options (keys are returned lower-case)."""
    lower = {n.lower() for n in names}
    rest, opts, i = [], {}, 0
    while i < len(args):
        a = args[i]
        if is_text(a) and text(a).lower() in lower and i + 1 < len(args):
            opts[text(a).lower()] = args[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    return rest, opts


def _read_text(fname: str) -> str:
    try:
        with open(fname, encoding="utf-8-sig", newline="") as fh:
            return fh.read()
    except FileNotFoundError:
        raise MatlabError(f"Unable to find or open '{fname}'. Check the path and filename or file permissions.")
    except OSError as e:
        raise MatlabError(f"Unable to read '{fname}': {e.strerror}.")


def _lines_of(content: str) -> list[str]:
    lines = re.split(r"\r\n|\n|\r", content)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _sniff_delimiter(lines: list[str]) -> str | None:
    """Guess the field delimiter; None means runs of whitespace."""
    sample = []
    for ln in lines:  # the first 50 non-blank lines
        if ln.strip():
            sample.append(ln)
            if len(sample) == 50:
                break
    for d in (",", "\t", ";", "|"):
        counts = [ln.count(d) for ln in sample[-20:]]
        if counts and min(counts) > 0:
            return d
    return None


def _split_fields(line: str, delim: str | None) -> list[str]:
    if delim is None:
        return line.split()
    if delim == " ":
        return [f for f in line.split(" ")]
    return next(csv.reader([line], delimiter=delim, skipinitialspace=True))


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*([eEdD][+-]?\d+)?|\.\d+([eEdD][+-]?\d+)?|inf|nan)$", re.IGNORECASE)


def _parse_number(s: str) -> float | None:
    t = s.strip()
    if t == "":
        return np.nan
    if _NUM_RE.match(t):
        return float(t.replace("d", "e").replace("D", "e"))
    if t.lower() in ("+inf", "-inf"):
        return float(t)
    return None


def _is_text_row(fields: list[str]) -> bool:
    nonempty = [f for f in fields if f.strip()]
    return bool(nonempty) and all(_parse_number(f) is None for f in nonempty)


def _numeric_table(lines: list[str], delim: str | None, header: int | None) -> tuple[np.ndarray, list[str], int]:
    """Parse delimited numeric text; returns (matrix, header lines, number of header lines)."""
    if not header:
        fast = _fastcsv.parse_numeric(lines, delim)  # nothing but plain numbers: convert in bulk
        if fast is not None:
            return fast, [], 0
    rows = [_split_fields(ln, delim) for ln in lines]
    if header is None:
        header = 0
        while header < len(rows) and (not lines[header].strip() or _is_text_row(rows[header])):
            header += 1
    body = [r for r, ln in zip(rows[header:], lines[header:]) if ln.strip()]
    width = max((len(r) for r in body), default=0)
    data = np.full((len(body), width), np.nan)
    for i, r in enumerate(body):
        for j, f in enumerate(r):
            v = _parse_number(f)
            data[i, j] = np.nan if v is None else v
    # a delimiter at the end of every line does not start another column
    if width and body and all(len(r) == width and r[-1].strip() == "" for r in body):
        data = data[:, :-1]
    return data, lines[:header], header


# -- numeric matrices ---------------------------------------------------------------------------

@builtin("readmatrix")
def _readmatrix(interp, args, nargout):
    nargs(args, 1, None, "readmatrix")
    from . import _textimport
    return _textimport.read_matrix(interp, list(args))


@builtin("csvread")
def _csvread(interp, args, nargout):
    nargs(args, 1, 4, "csvread")
    data, _, _ = _numeric_table(_lines_of(_read_text(text(args[0]))), ",", 0)
    data = np.where(np.isnan(data), 0.0, data)
    return _offset(data, args[1:])


@builtin("dlmread")
def _dlmread(interp, args, nargout):
    nargs(args, 1, 4, "dlmread")
    lines = _lines_of(_read_text(text(args[0])))
    rest = list(args[1:])
    delim = _delimiter(rest.pop(0)) if rest and is_text(rest[0]) else _sniff_delimiter(lines)
    data, _, _ = _numeric_table(lines, delim, 0)
    data = np.where(np.isnan(data), 0.0, data)
    return _offset(data, rest)


def _offset(data: np.ndarray, rest: list) -> np.ndarray:
    if len(rest) >= 2:
        r0, c0 = to_int(rest[0], "row offset"), to_int(rest[1], "column offset")
        data = data[r0:, c0:]
    elif len(rest) == 1:
        rng = num(rest[0]).flatten().astype(int)
        if rng.size == 4:
            data = data[rng[0]:rng[2] + 1, rng[1]:rng[3] + 1]
    return data


def _format_number(x, fmt: str | None) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, complex) or isinstance(x, np.complexfloating):
        re_, im = _format_number(x.real, fmt), _format_number(abs(x.imag), fmt)
        return f"{re_}{'-' if x.imag < 0 else '+'}{im}i"
    xf = float(x)
    if np.isnan(xf):
        return "NaN"
    if np.isinf(xf):
        return "Inf" if xf > 0 else "-Inf"
    if fmt is not None:
        return fmt % xf
    return f"{xf:.15g}"


def _quote(s: str, delim: str) -> str:
    if delim in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _cell_text(x, delim: str, fmt: str | None) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return _quote(x, delim)
    if isinstance(x, (MString, StringArray)) or is_char(x):
        if isinstance(x, StringArray):
            v = x.data.flat[0] if x.data.size else ""
            return "" if v is None else _quote(v, delim)
        return _quote(to_str(x), delim)
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        if x.size == 1:
            return _format_number(x.flat[0], fmt)
        return " ".join(_format_number(v, fmt) for v in x.flatten(order="F"))
    raise MatlabError("Cells to be written must hold numbers or text.")


def _write_rows(fname: str, rows: list[list[str]], delim: str, append: bool, newline: str = "\n") -> None:
    try:
        with open(fname, "a" if append else "w", encoding="utf-8", newline="") as fh:
            for r in rows:
                fh.write(delim.join(r) + newline)
    except OSError as e:
        raise MatlabError(f"Unable to write to '{fname}': {e.strerror}.")


def _write_matrix(fname: str, a, delim: str, fmt: str | None, append: bool, newline: str = "\n") -> None:
    """Write a matrix (or cell/string array) as delimited text."""
    if not _fastcsv.write_numeric(fname, a, delim, fmt, append, newline):
        _write_rows(fname, _matrix_rows(a, delim, fmt), delim, append, newline)


def _matrix_rows(a, delim: str, fmt: str | None) -> list[list[str]]:
    if isinstance(a, Cell):
        return [[_cell_text(a.data[i, j], delim, fmt) for j in range(a.data.shape[1])]
                for i in range(a.data.shape[0])]
    if isinstance(a, StringArray):
        return [[_cell_text(a.data[i, j], delim, fmt) for j in range(a.data.shape[1])]
                for i in range(a.data.shape[0])]
    if isinstance(a, MString):
        return [[_quote(str(a), delim)]]
    if is_char(a):
        return [[_quote("".join(row), delim)] for row in a]
    m = a if isinstance(a, np.ndarray) else num(a)
    if m.ndim > 2:
        m = m.reshape(m.shape[0], -1, order="F")
    return [[_format_number(v, fmt) for v in row] for row in m]


def _default_delim(fname: str) -> str:
    ext = os.path.splitext(fname)[1].lower()
    return "\t" if ext == ".tsv" else ","


@builtin("writematrix")
def _writematrix(interp, args, nargout):
    nargs(args, 1, None, "writematrix")
    a = args[0]
    rest = list(args[1:])
    fname = text(rest.pop(0)) if rest and is_text(rest[0]) and \
        text(rest[0]).lower() not in ("delimiter", "writemode", "filetype") else "matrix.txt"
    _, opts = _name_value(rest, {"Delimiter", "WriteMode", "FileType", "QuoteStrings"})
    delim = _delimiter(opts["delimiter"]) if "delimiter" in opts else _default_delim(fname)
    append = "writemode" in opts and text(opts["writemode"]).lower() == "append"
    _write_matrix(fname, a, delim, None, append)
    return None


@builtin("csvwrite")
def _csvwrite(interp, args, nargout):
    nargs(args, 2, 4, "csvwrite")
    fname, a = text(args[0]), num(args[1])
    a = _pad_offset(a, args[2:])
    _write_matrix(fname, a, ",", "%.5g", False)
    return None


def _pad_offset(a: np.ndarray, rest: list) -> np.ndarray:
    if len(rest) >= 2:
        r, c = to_int(rest[0], "row offset"), to_int(rest[1], "column offset")
        out = np.zeros((a.shape[0] + r, a.shape[1] + c))
        out[r:, c:] = a
        return out
    return a


@builtin("dlmwrite")
def _dlmwrite(interp, args, nargout):
    nargs(args, 2, None, "dlmwrite")
    fname, a = text(args[0]), args[1]
    rest = list(args[2:])
    append = False
    if any(is_text(x) and text(x) == "-append" for x in rest):
        append = True
        rest = [x for x in rest if not (is_text(x) and text(x) == "-append")]
    delim = ","
    if rest and is_text(rest[0]) and text(rest[0]).lower() not in ("delimiter", "precision", "newline",
                                                                   "roffset", "coffset"):
        delim = _delimiter(rest.pop(0))
        if len(rest) >= 2 and not is_text(rest[0]):
            a = _pad_offset(num(a), rest[:2])
            rest = rest[2:]
    _, opts = _name_value(rest, {"delimiter", "precision", "newline", "roffset", "coffset"})
    if "delimiter" in opts:
        delim = _delimiter(opts["delimiter"])
    fmt = "%.5g"
    if "precision" in opts:
        p = opts["precision"]
        fmt = text(p) if is_text(p) else f"%.{to_int(p, 'precision')}g"
    newline = "\n"
    if "newline" in opts and text(opts["newline"]).lower() == "pc":
        newline = "\r\n"
    if "roffset" in opts or "coffset" in opts:
        r = to_int(opts.get("roffset", np.array([[0.0]])), "roffset")
        c = to_int(opts.get("coffset", np.array([[0.0]])), "coffset")
        a = _pad_offset(num(a), [np.array([[float(r)]]), np.array([[float(c)]])])
    _write_matrix(fname, a if not is_char(a) else num(a), delim, fmt, append, newline)
    return None


# -- cells and lines ------------------------------------------------------------------------------

@builtin("writecell")
def _writecell(interp, args, nargout):
    nargs(args, 1, None, "writecell")
    c = args[0]
    if not isinstance(c, Cell):
        raise MatlabError("writecell expects a cell array.")
    rest = list(args[1:])
    fname = text(rest.pop(0)) if rest and is_text(rest[0]) and \
        text(rest[0]).lower() not in ("delimiter", "writemode", "filetype") else "cell.txt"
    _, opts = _name_value(rest, {"Delimiter", "WriteMode", "FileType", "QuoteStrings"})
    delim = _delimiter(opts["delimiter"]) if "delimiter" in opts else _default_delim(fname)
    append = "writemode" in opts and text(opts["writemode"]).lower() == "append"
    _write_rows(fname, _matrix_rows(c, delim, None), delim, append)
    return None


@builtin("readcell")
def _readcell(interp, args, nargout):
    nargs(args, 1, None, "readcell")
    from . import _textimport
    return _textimport.read_cell(interp, list(args))


@builtin("readlines")
def _readlines(interp, args, nargout):
    nargs(args, 1, None, "readlines")
    _, opts = _name_value(list(args[1:]), {"EmptyLineRule", "WhitespaceRule", "LineEnding", "Encoding"})
    lines = _lines_of(_read_text(text(args[0])))
    rule = text(opts["emptylinerule"]).lower() if "emptylinerule" in opts else "read"
    ws = text(opts["whitespacerule"]).lower() if "whitespacerule" in opts else "preserve"
    if ws == "trim":
        lines = [ln.strip() for ln in lines]
    elif ws == "trimleading":
        lines = [ln.lstrip() for ln in lines]
    elif ws == "trimtrailing":
        lines = [ln.rstrip() for ln in lines]
    if rule == "skip":
        lines = [ln for ln in lines if ln.strip()]
    data = np.empty((len(lines), 1), dtype=object)
    for i, ln in enumerate(lines):
        data[i, 0] = ln
    return string_result(data) if len(lines) == 1 else StringArray(data)


@builtin("writelines")
def _writelines(interp, args, nargout):
    nargs(args, 2, None, "writelines")
    v, fname = args[0], text(args[1])
    _, opts = _name_value(list(args[2:]), {"WriteMode", "LineEnding", "Encoding"})
    if isinstance(v, StringArray):
        lines = ["" if x is None else x for x in v.data.flatten(order="F")]
    elif isinstance(v, Cell):
        lines = [to_str(x) for x in v.values_f()]
    elif is_char(v) and v.ndim == 2 and v.shape[0] > 1:
        lines = ["".join(r) for r in v]
    else:
        lines = [text(v)]
    ending = "\n"
    if "lineending" in opts:
        ending = {"\\n": "\n", "\\r\\n": "\r\n", "\\r": "\r"}.get(text(opts["lineending"]), text(opts["lineending"]))
    append = "writemode" in opts and text(opts["writemode"]).lower() == "append"
    try:
        with open(fname, "a" if append else "w", encoding="utf-8", newline="") as fh:
            for ln in lines:
                fh.write(ln + ending)
    except OSError as e:
        raise MatlabError(f"Unable to write to '{fname}': {e.strerror}.")
    return None


@builtin("type")
def _type(interp, args, nargout):
    nargs(args, 1, None, "type")
    for a in args:
        name = text(a)
        path = name
        if not os.path.isfile(path):
            candidates = [name + ".m"] + [os.path.join(p, name) for p in getattr(interp, "path", [])] + \
                [os.path.join(p, name + ".m") for p in getattr(interp, "path", [])]
            path = next((c for c in candidates if os.path.isfile(c)), None)
            if path is None:
                raise MatlabError(f"File '{name}' not found.")
        content = _read_text(path)
        if nargout > 0:
            return char(content)
        interp.out("\n" + content + ("" if content.endswith("\n") else "\n") + "\n")
    return None


# -- importdata -------------------------------------------------------------------------------

@builtin("importdata")
def _importdata(interp, args, nargout):
    nargs(args, 1, 3, "importdata")
    fname = text(args[0])
    if fname.lower().endswith(".mat"):
        from ._registry import BUILTINS
        res = BUILTINS["load"](interp, [char(fname)], 1)
        return res
    lines = _lines_of(_read_text(fname))
    delim = _delimiter(args[1]) if len(args) > 1 and is_text(args[1]) else _sniff_delimiter(lines)
    header = to_int(args[2], "header lines") if len(args) > 2 else None
    data, head, nhead = _numeric_table(lines, delim, header)
    out_delim = char(delim if delim is not None else " ")
    if nhead == 0:
        result = data
    else:
        last = _split_fields(head[-1], delim) if head else []
        fields: dict = {"data": data}
        if head and data.shape[1] and len(last) == data.shape[1] and all(f.strip() for f in last):
            colheaders = Cell.from_list([char(f.strip()) for f in last])
            if nhead == 1:
                fields["textdata"] = colheaders
            else:
                fields["textdata"] = Cell.from_list([char(h) for h in head], column=True)
            fields["colheaders"] = colheaders
        else:
            fields["textdata"] = Cell.from_list([char(h) for h in head], column=True)
        result = Struct.scalar(fields)
    outs = (result, out_delim, float(nhead))
    return outs[:nargout] if nargout > 1 else result


# -- scanning formatted text ----------------------------------------------------------------------

@builtin("fscanf")
def _fscanf(interp, args, nargout):
    nargs(args, 2, 3, "fscanf")
    fh = file_handle(interp, to_int(args[0], "file identifier"))
    content = fh.read()
    from ._registry import BUILTINS
    res = BUILTINS["sscanf"](interp, [char(content), *args[1:]], max(nargout, 1))
    if nargout > 1 and isinstance(res, tuple):
        return res[:2]
    return res


@builtin("textscan")
def _textscan(interp, args, nargout):
    """Read formatted text from a file or a string (the scanner is in _textimport)."""
    nargs(args, 1, None, "textscan")
    from . import _textimport
    return _textimport.textscan(interp, list(args), nargout)


# -- folders --------------------------------------------------------------------------------------

def _datenum(ts: float) -> float:
    d = _dt.datetime.fromtimestamp(ts)
    midnight = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return d.toordinal() + 366 + (d - midnight).total_seconds() / 86400.0


def _dir_entry(path: str, name: str) -> dict:
    full = os.path.join(path, name)
    try:
        st = os.stat(full)
        mtime = st.st_mtime
        size = st.st_size
    except OSError:
        mtime, size = 0.0, 0
    isdir = os.path.isdir(full)
    stamp = _dt.datetime.fromtimestamp(mtime).strftime("%d-%b-%Y %H:%M:%S")
    return {
        "name": char(name), "folder": char(os.path.abspath(path)), "date": char(stamp),
        "bytes": np.array([[0.0 if isdir else float(size)]]), "isdir": np.array([[isdir]]),
        "datenum": np.array([[_datenum(mtime)]]),
    }


_DIR_FIELDS = ["name", "folder", "date", "bytes", "isdir", "datenum"]


@builtin("dir")
def _dir(interp, args, nargout):
    nargs(args, 0, 1, "dir")
    target = text(args[0]) if args else "."
    target = os.path.expanduser(target)
    entries: list[dict] = []
    if any(ch in target for ch in "*?["):
        for match in sorted(glob.glob(target)):
            folder, name = os.path.split(match)
            entries.append(_dir_entry(folder or ".", name))
    elif os.path.isdir(target):
        names = [".", ".."] + sorted(os.listdir(target), key=str.lower)
        entries = [_dir_entry(target, n) for n in names]
    elif os.path.exists(target):
        folder, name = os.path.split(target)
        entries.append(_dir_entry(folder or ".", name))
    if nargout == 0:
        if not entries:
            interp.out(f"No files or folders match '{text(args[0]) if args else '.'}'.\n" if args else "")
            return None
        names = [to_str(e["name"]) for e in entries]
        width = max(len(n) for n in names) + 2
        per_line = max(1, 80 // width)
        lines = []
        for i in range(0, len(names), per_line):
            lines.append("".join(n.ljust(width) for n in names[i:i + per_line]).rstrip())
        interp.out("\n" + "\n".join(lines) + "\n\n")
        return None
    data = np.empty((len(entries), 1), dtype=object)
    for i, e in enumerate(entries):
        data[i, 0] = e
    return Struct(data, list(_DIR_FIELDS))


@builtin("tempdir")
def _tempdir(interp, args, nargout):
    import tempfile
    return char(os.path.join(tempfile.gettempdir(), ""))


@builtin("tempname")
def _tempname(interp, args, nargout):
    import uuid
    import tempfile
    folder = text(args[0]) if args else tempfile.gettempdir()
    return char(os.path.join(folder, "tp" + uuid.uuid4().hex))


@builtin("rmdir")
def _rmdir(interp, args, nargout):
    nargs(args, 1, 2, "rmdir")
    path = text(args[0])
    recursive = len(args) > 1 and text(args[1]).lower() == "s"
    msg = ""
    try:
        if recursive:
            shutil.rmtree(path)
        else:
            os.rmdir(path)
        ok = True
    except FileNotFoundError:
        ok, msg = False, f"'{path}' is not a folder or does not exist."
    except OSError as e:
        ok, msg = False, (f"Cannot remove '{path}': the folder is not empty; use rmdir(folder, 's')."
                          if not recursive and os.path.isdir(path) and os.listdir(path) else
                          f"Cannot remove '{path}': {e.strerror}.")
    if nargout == 0:
        if not ok:
            raise MatlabError(msg)
        return None
    outs = (bool_scalar(ok), char(msg), char("" if ok else "mylab:rmdir:failed"))
    return outs[:nargout] if nargout > 1 else outs[0]


@builtin("textread")
def _textread(interp, args, nargout):
    """Read a whole text file by format; one output per conversion."""
    nargs(args, 1, None, "textread")
    fname = text(args[0])
    path = fname
    if not os.path.isfile(path):
        path = next((c for c in (os.path.join(d, fname) for d in getattr(interp, "path", []))
                     if os.path.isfile(c)), fname)
    rest = list(args[1:])
    fmt = ""
    if rest and is_text(rest[0]) and ("%" in text(rest[0]) or text(rest[0]) == ""):
        fmt = text(rest.pop(0))
    content = _read_text(path)
    if fmt == "":
        _, opts = _name_value(rest, {"delimiter", "headerlines"})
        lines = _lines_of(content)
        skip = to_int(opts["headerlines"], "headerlines") if "headerlines" in opts else 0
        delim = _delimiter(opts["delimiter"]) if "delimiter" in opts else _sniff_delimiter(lines[skip:])
        data, _, _ = _numeric_table(lines[skip:], delim or None, 0)
        data = np.where(np.isnan(data), 0.0, data)
        if nargout > 1:
            return tuple(data[:, [j]] for j in range(min(nargout, data.shape[1])))
        return data
    return _scan_outputs(interp, content, fmt, rest, nargout, "textread")


def _scan_outputs(interp, content: str, fmt: str, rest: list, nargout: int, name: str):
    cells = _textscan(interp, [char(content), char(fmt), *rest], 1)
    # unlike textscan, these older functions return every number as double
    outs = tuple(v.astype(float) if isinstance(v, np.ndarray) and v.dtype.kind in "iu" else v
                 for v in cells.data.flat)
    if nargout > len(outs):
        raise MatlabError(f"{name}: more outputs requested than conversions in the format.")
    if nargout <= 1:
        return outs[0] if outs else np.zeros((0, 0))
    return outs[:nargout]


@builtin("strread")
def _strread(interp, args, nargout):
    """Like textread, but the text comes from the first argument instead of a file."""
    nargs(args, 1, None, "strread")
    rest = list(args[1:])
    fmt = text(rest.pop(0)) if rest and is_text(rest[0]) and "%" in text(rest[0]) else "%f"
    return _scan_outputs(interp, text(args[0], "input text"), fmt, rest, nargout, "strread")

