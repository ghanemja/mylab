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
from ._registry import builtin
from ._util import bool_scalar, fnum, is_text, nargs, num, text
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
    sample = [ln for ln in lines if ln.strip()][:50]
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
    fname = text(args[0])
    _, opts = _name_value(list(args[1:]), {"NumHeaderLines", "Delimiter", "Range", "FileType",
                                           "OutputType", "ExpectedNumVariables"})
    lines = _lines_of(_read_text(fname))
    delim = _delimiter(opts["delimiter"]) if "delimiter" in opts else _sniff_delimiter(lines)
    header = to_int(opts["numheaderlines"], "NumHeaderLines") if "numheaderlines" in opts else None
    data, _, _ = _numeric_table(lines, delim, header)
    if "range" in opts and not is_text(opts["range"]):
        r = num(opts["range"]).flatten().astype(int)
        if r.size >= 2:
            data = data[r[0] - 1:, r[1] - 1:] if r.size == 2 else data[r[0] - 1:r[2], r[1] - 1:r[3]]
    return data


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
    _write_rows(fname, _matrix_rows(a, delim, None), delim, append)
    return None


@builtin("csvwrite")
def _csvwrite(interp, args, nargout):
    nargs(args, 2, 4, "csvwrite")
    fname, a = text(args[0]), num(args[1])
    a = _pad_offset(a, args[2:])
    _write_rows(fname, _matrix_rows(a, ",", "%.5g"), ",", False)
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
    _write_rows(fname, _matrix_rows(a if not is_char(a) else num(a), delim, fmt), delim, append, newline)
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


def _missing_value():
    d = np.empty((1, 1), dtype=object)
    d[0, 0] = None
    return StringArray(d)


@builtin("readcell")
def _readcell(interp, args, nargout):
    nargs(args, 1, None, "readcell")
    fname = text(args[0])
    _, opts = _name_value(list(args[1:]), {"Delimiter", "NumHeaderLines", "FileType"})
    lines = _lines_of(_read_text(fname))
    delim = _delimiter(opts["delimiter"]) if "delimiter" in opts else _sniff_delimiter(lines)
    skip = to_int(opts["numheaderlines"], "NumHeaderLines") if "numheaderlines" in opts else 0
    rows = [_split_fields(ln, delim) for ln in lines[skip:] if ln.strip()]
    width = max((len(r) for r in rows), default=0)
    data = np.empty((len(rows), width), dtype=object)
    for i, r in enumerate(rows):
        for j in range(width):
            f = r[j].strip() if j < len(r) else ""
            if f == "":
                data[i, j] = _missing_value()
                continue
            v = _parse_number(f)
            data[i, j] = np.array([[v]]) if v is not None else char(f)
    return Cell(data)


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


_TS_SPEC = re.compile(r"%(\*?)(\d*)(?:\.(\d+))?(f|d|u|n|s|q|c|\[[^\]]*\])(8|16|32|64)?")


def _parse_ts_format(fmt: str) -> list[tuple[str, bool, str, str]]:
    """List of (kind, skip, bits, literal-prefix) for each conversion in a textscan format."""
    specs = []
    pos = 0
    for m in _TS_SPEC.finditer(fmt):
        literal = fmt[pos:m.start()].strip()
        specs.append((m.group(4), bool(m.group(1)), m.group(5) or "", literal))
        pos = m.end()
    if not specs:
        raise MatlabError("textscan: the format must contain at least one conversion such as %f or %s.")
    return specs


@builtin("textscan")
def _textscan(interp, args, nargout):
    nargs(args, 1, None, "textscan")
    src = args[0]
    rest = list(args[1:])
    fmt = text(rest.pop(0)) if rest and is_text(rest[0]) and text(rest[0]).lower() not in (
        "delimiter", "headerlines") else "%f"
    from .strings import process_escapes
    fmt = process_escapes(fmt)
    repeat = None
    if rest and not is_text(rest[0]):
        r = float(fnum(rest.pop(0)).flat[0])
        repeat = None if np.isinf(r) else int(r)
    _, opts = _name_value(rest, {"Delimiter", "HeaderLines", "EndOfLine", "MultipleDelimsAsOne",
                                 "CollectOutput", "Whitespace", "CommentStyle", "ReturnOnError",
                                 "EmptyValue", "TextType"})
    fh = None
    if is_text(src):
        content = text(src)
    else:
        fh = file_handle(interp, to_int(src, "file identifier"))
        content = fh.read()
    lines = _lines_of(content)
    skip = to_int(opts["headerlines"], "HeaderLines") if "headerlines" in opts else 0
    lines = lines[skip:]
    if "commentstyle" in opts:
        cs = text(opts["commentstyle"])
        lines = [ln.split(cs, 1)[0] for ln in lines]
    delims: list[str] | None = None
    if "delimiter" in opts:
        d = opts["delimiter"]
        items = [to_str(x) for x in d.values_f()] if isinstance(d, Cell) else [text(d)]
        delims = [process_escapes(x) for x in items]
    collapse = "multipledelimsasone" in opts and bool(num(opts["multipledelimsasone"]).flat[0])
    empty_value = float(fnum(opts["emptyvalue"]).flat[0]) if "emptyvalue" in opts else np.nan
    as_string = "texttype" in opts and text(opts["texttype"]).lower() == "string"
    specs = _parse_ts_format(fmt)
    tokens: list[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if delims is None:
            tokens.extend(_ts_split_ws(ln))
        else:
            pattern = "|".join(re.escape(x) for x in sorted(delims, key=len, reverse=True))
            if collapse:
                pattern = f"(?:{pattern})+"
            parts = re.split(pattern, ln)
            tokens.extend(p.strip(" \t") for p in parts)
    columns: list[list] = [[] for _ in specs]
    k = 0
    records = 0
    stop = False
    while k < len(tokens) and not stop:
        if repeat is not None and records >= repeat:
            break
        for j, (kind, skip_it, _bits, literal) in enumerate(specs):
            if k >= len(tokens):
                break
            tok = tokens[k]
            if literal and tok.startswith(literal):
                tok = tok[len(literal):]
            if kind in ("f", "d", "u", "n"):
                v = _parse_number(tok)
                if v is None:
                    stop = True
                    break
                if tok.strip() == "":
                    v = empty_value
                if not skip_it:
                    columns[j].append(v)
            elif kind == "q":
                t = tok.strip()
                if len(t) >= 2 and t[0] == t[-1] == '"':
                    t = t[1:-1].replace('""', '"')
                if not skip_it:
                    columns[j].append(t)
            else:
                if not skip_it:
                    columns[j].append(tok)
            k += 1
        else:
            records += 1
            continue
        break
    out = []
    for (kind, skip_it, bits, _lit), col in zip(specs, columns):
        if skip_it:
            continue
        if kind in ("f", "n"):
            arr = np.array(col, dtype=np.float32 if bits == "32" and kind == "f" else float).reshape(-1, 1)
            out.append(arr)
        elif kind in ("d", "u"):
            dtype = {"8": np.int8, "16": np.int16, "32": np.int32, "64": np.int64}.get(bits, np.int32)
            if kind == "u":
                dtype = {"8": np.uint8, "16": np.uint16, "32": np.uint32, "64": np.uint64}.get(bits, np.uint32)
            vals = np.array(col, dtype=float)
            info = np.iinfo(dtype)
            vals = np.clip(np.where(np.isnan(vals), 0, np.round(vals)), info.min, info.max)
            out.append(vals.astype(dtype).reshape(-1, 1))
        elif kind == "c":
            out.append(np.array([[c[0] if c else " "] for c in col], dtype="<U1").reshape(-1, 1))
        else:
            if as_string:
                d = np.empty((len(col), 1), dtype=object)
                for i, x in enumerate(col):
                    d[i, 0] = x
                out.append(StringArray(d))
            else:
                out.append(Cell.from_list([char(x) for x in col], column=True))
    if "collectoutput" in opts and bool(num(opts["collectoutput"]).flat[0]):
        merged: list = []
        for v in out:
            if merged and isinstance(v, np.ndarray) and isinstance(merged[-1], np.ndarray) and \
                    v.dtype == merged[-1].dtype and v.shape[0] == merged[-1].shape[0]:
                merged[-1] = np.hstack([merged[-1], v])
            elif merged and isinstance(v, Cell) and isinstance(merged[-1], Cell) and \
                    v.data.shape[0] == merged[-1].data.shape[0]:
                merged[-1] = Cell(np.hstack([merged[-1].data, v.data]))
            else:
                merged.append(v)
        out = merged
    result = Cell.from_list(out)
    if nargout > 1:
        return result, float(len(content))
    return result


def _ts_split_ws(line: str) -> list[str]:
    """Whitespace-separated tokens, keeping "double quoted" text together."""
    out = []
    for m in re.finditer(r'"(?:[^"]|"")*"|\S+', line):
        out.append(m.group(0))
    return out


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

