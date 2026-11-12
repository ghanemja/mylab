"""Text: formatting, conversion, searching and splitting."""

from __future__ import annotations

import math
import re

import numpy as np

from ..errors import MatlabError
from ..values import (
    Cell, MString, StringArray, char, is_char, string_data, string_result, to_int, to_str,
)

from ._registry import builtin
from ._util import bool_scalar, fnum, is_text, nargs, num, text


def _as_cell_if_strings(v):
    """Treat a string array like a cell array of char vectors (same shape)."""
    if isinstance(v, StringArray):
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            x = v.data[idx]
            data[idx] = char(x) if x is not None else char("")
        return Cell(data)
    return v

# -- sprintf ----------------------------------------------------------------------------

_SPEC = re.compile(r"%(?P<flags>[-+ 0#]*)(?P<width>\*|\d+)?(?:\.(?P<prec>\*|\d+))?(?:hh|h|ll|l|L|j|z|t)?"
                   r"(?P<conv>[diouxXfeEgGcs%])")

_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def process_escapes(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
            if nxt == "x":
                m = re.match(r"[0-9a-fA-F]+", s[i + 2:])
                if m:
                    out.append(chr(int(m.group(0), 16)))
                    i += 2 + len(m.group(0))
                    continue
            if nxt in "01234567":
                m = re.match(r"[0-7]{1,3}", s[i + 1:])
                out.append(chr(int(m.group(0), 8)))
                i += 1 + len(m.group(0))
                continue
        out.append(ch)
        i += 1
    return "".join(out)


class _StringItem(str):
    """Text from a string scalar: always printed whole (a char vector is split per character
    when a numeric conversion consumes it)."""


def _flatten_args(args) -> list:
    items: list = []
    for a in args:
        if isinstance(a, MString):
            items.append(_StringItem(str(a)))
        elif is_char(a):
            items.append("".join(a.flatten(order="F")))
        elif isinstance(a, StringArray):
            items.extend(_StringItem("" if x is None else x) for x in a.data.flatten(order="F"))
        elif isinstance(a, Cell):
            raise MatlabError("sprintf/fprintf do not accept cell arrays; use c{:} to pass contents.")
        elif isinstance(a, np.ndarray):
            flat = a.flatten(order="F")
            if a.dtype == bool:
                items.extend(float(x) for x in flat)
            elif a.dtype.kind in "iu":
                items.extend(int(x) for x in flat)
            else:
                items.extend(flat.tolist())
        else:
            raise MatlabError("Unsupported argument type for formatting.")
    return items


def _fmt_nonfinite(x: float, flags: str, width: str | None) -> str:
    s = "NaN" if math.isnan(x) else ("Inf" if x > 0 else "-Inf")
    if "+" in flags and s == "Inf":
        s = "+Inf"
    w = int(width) if width and width != "*" else 0
    return s.ljust(w) if "-" in flags else s.rjust(w)


def _format_one(spec: re.Match, value, width, prec) -> str:
    flags = spec.group("flags") or ""
    conv = spec.group("conv")
    w = "" if width is None else str(width)
    p = "" if prec is None else f".{prec}"
    if conv != "s" and isinstance(value, str) and not isinstance(value, _StringItem):
        if conv == "c":
            return f"%{flags}{w}s" % value
        value = float(ord(value[0])) if value else 0.0
    if conv == "s":
        if isinstance(value, str):
            return f"%{flags}{w}{p}s" % value
        if isinstance(value, complex):
            value = value.real
        if float(value).is_integer() and math.isfinite(value):
            if 0 <= value <= 0x10FFFF:
                # an integer consumed by %s is taken as a character code
                return f"%{flags}{w}s" % chr(int(value))
            return f"%{flags}{w}s" % _num_text(value)
        if not math.isfinite(value):
            return _fmt_nonfinite(value, flags, w)
        return f"%{flags}{w}{p or ''}e" % value if p else f"%{flags}{w}s" % _shortg(value)
    if isinstance(value, str):
        # numeric conversion fed a string: MATLAB prints it as text
        return f"%{flags}{w}s" % value
    if isinstance(value, complex):
        value = value.real
    if conv == "c":
        return chr(int(value)) if float(value).is_integer() else _shortg(value)
    if isinstance(value, float) and not math.isfinite(value):
        return _fmt_nonfinite(value, flags, w)
    if conv in "diu":
        if float(value) != int(value):
            # non-integer with %d: MATLAB switches to %e
            return f"%{flags}{w}{p or ''}e" % value
        return f"%{flags}{w}{p}d" % int(value)
    if conv in "oxX":
        if float(value) != int(value):
            return f"%{flags}{w}{p or ''}e" % value
        if conv == "o" and "#" in flags:
            # C style: a leading zero, not Python's 0o
            digits = "%o" % abs(int(value))
            body = ("-" if value < 0 else "") + ("0" + digits if digits != "0" else "0")
            return f"%{flags.replace('#', '')}{w}s" % body
        return f"%{flags}{w}{p}{conv}" % int(value)
    return f"%{flags}{w}{p}{conv}" % float(value)


def _num_text(x: float) -> str:
    return str(int(x)) if abs(x) < 1e16 else f"{x:g}"


def _shortg(x: float) -> str:
    return f"{x:e}"


def matlab_sprintf(fmt: str, args) -> str:
    fmt = process_escapes(fmt)
    items = _flatten_args(args)
    specs = [m for m in _SPEC.finditer(fmt)]
    data_specs = [m for m in specs if m.group("conv") != "%"]
    out: list[str] = []
    if not data_specs:
        text_ = _SPEC.sub(lambda m: "%" if m.group("conv") == "%" else m.group(0), fmt)
        return text_
    pos = 0
    first = True
    while first or pos < len(items):
        first = False
        last = 0
        consumed_any = False
        for m in specs:
            out.append(fmt[last:m.start()])
            last = m.end()
            if m.group("conv") == "%":
                out.append("%")
                continue
            width = m.group("width")
            prec = m.group("prec")
            if width == "*":
                if pos >= len(items):
                    return "".join(out)
                width = int(items[pos])
                pos += 1
            if prec == "*":
                if pos >= len(items):
                    return "".join(out)
                prec = int(items[pos])
                pos += 1
            if pos >= len(items):
                if items:
                    return "".join(out)
                # no data at all: print literal text, skipping the conversion
                continue
            value = items[pos]
            if (m.group("conv") != "s" and isinstance(value, str) and len(value) > 1
                    and not isinstance(value, _StringItem)):
                # a char vector feeds a non-%s conversion one character at a time
                items[pos] = value[1:]
                value = value[0]
            else:
                pos += 1
            consumed_any = True
            out.append(_format_one(m, value, width, prec))
        out.append(fmt[last:])
        if not consumed_any:
            break
    return "".join(out)


@builtin("sprintf")
def _sprintf(interp, args, nargout):
    nargs(args, 1, None, "sprintf")
    s = matlab_sprintf(text(args[0], "format"), args[1:])
    return MString(s) if isinstance(args[0], MString) else char(s)


@builtin("fprintf")
def _fprintf(interp, args, nargout):
    nargs(args, 1, None, "fprintf")
    fid = None
    if not is_text(args[0]):
        fid = to_int(args[0], "file identifier")
        args = args[1:]
    s = matlab_sprintf(text(args[0], "format"), args[1:])
    if fid is None or fid == 1 or fid == 2:
        interp.out(s)
    else:
        from .system import file_handle
        file_handle(interp, fid).write(s)
    if nargout > 0:
        return float(len(s.encode()))
    return None


# -- number <-> text ---------------------------------------------------------------------

def _num2str_scalar(x, fmt=None, digits=None) -> str:
    if isinstance(x, (complex, np.complexfloating)):
        re_, im = float(np.real(x)), float(np.imag(x))
        sign = "-" if im < 0 else "+"
        return f"{_num2str_scalar(re_, fmt, digits)}{sign}{_num2str_scalar(abs(im), fmt, digits)}i"
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    x = float(x)
    if fmt is not None:
        return matlab_sprintf(fmt, [np.array([[x]])])
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if digits is not None:
        return f"{x:.{digits}g}"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    sig = max(int(math.floor(math.log10(abs(x)))) + 5, 5) if x != 0 else 5
    return f"{x:.{sig}g}"


_BLANKS = " \t\n\r\f\v\0"


def _trim_rows(rows: list[str], leading: bool = True) -> list[str]:
    """Remove the blank columns that all rows share (strtrim / deblank of a char matrix)."""
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    rows = [r.ljust(width) for r in rows]
    trail = min(len(r) - len(r.rstrip(_BLANKS)) for r in rows)
    lead = min(len(r) - len(r.lstrip(_BLANKS)) for r in rows) if leading else 0
    if lead >= width:
        return ["" for _ in rows]
    return [r[lead:width - trail] for r in rows]


def _rows_to_char(rows: list[str]) -> np.ndarray:
    width = max((len(r) for r in rows), default=0)
    return np.array([list(r.ljust(width)) for r in rows], dtype="<U1").reshape(len(rows), width)


def _g_field(x: float, width: int, prec: int) -> str:
    if math.isnan(x):
        return "NaN".rjust(width)
    if math.isinf(x):
        return ("Inf" if x > 0 else "-Inf").rjust(width)
    return "%*.*g" % (width, prec, x + 0.0)


def _num2str_real(a: np.ndarray, digits) -> list[str]:
    """Rows of text for a real 2-D array (before the common blanks are trimmed)."""
    scalar = a.size == 1
    if a.dtype.kind in "biu":
        cells = [[str(int(x)) for x in row] for row in a]
        width = 0 if scalar else max(len(c.lstrip("-")) for r in cells for c in r) + 2
        return ["".join(c.rjust(width) for c in r) for r in cells]
    a = a.astype(float)
    finite = np.isfinite(a)
    top = float(np.max(np.abs(a[finite]))) if finite.any() else 0.0
    if digits is not None:
        width = 0 if scalar else digits + 7
        return ["".join(_g_field(float(x), width, digits) for x in row) for row in a]
    if bool(np.all(a[finite] == np.round(a[finite]))) and top < 1e16:
        ndig = len(str(int(top)))
        if not finite.all():
            ndig = max(ndig, 3)
        width = 0 if scalar else ndig + 2
        return ["".join(_g_field(float(x), width, 16) for x in row) for row in a]
    ndig = int(math.floor(math.log10(top))) + 1 if top > 0 else 1
    prec = min(max(ndig + 4, 5), 16)
    width = 0 if scalar else prec + 7
    return ["".join(_g_field(float(x), width, prec) for x in row) for row in a]


@builtin("num2str")
def _num2str(interp, args, nargout):
    nargs(args, 1, 2, "num2str")
    v = args[0]
    if is_text(v):
        return v if is_char(v) else char(str(v))
    a = num(v)
    fmt = None
    digits = None
    if len(args) > 1:
        if is_text(args[1]):
            fmt = text(args[1])
        else:
            digits = to_int(args[1], "precision")
    if a.size == 0:
        return char("")
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1, order="F")
    if fmt is not None:
        rows = [matlab_sprintf(fmt, [row.reshape(1, -1)]) for row in a]
    elif a.dtype.kind == "c":
        if a.size == 1:
            return char(_num2str_scalar(a.flat[0], None, digits))
        cells = [[_num2str_scalar(x, None, digits) for x in row] for row in a]
        width = max(len(c) for r in cells for c in r) + 2
        rows = ["".join(c.rjust(width) for c in r) for r in cells]
    else:
        rows = _num2str_real(a, digits)
    rows = _trim_rows(rows)
    if len(rows) == 1:
        return char(rows[0])
    return _rows_to_char(rows)


@builtin("int2str")
def _int2str(interp, args, nargout):
    v = args[0]
    if isinstance(v, np.ndarray) and v.dtype.kind in "biu":
        return _num2str(interp, [v], 1)
    a = fnum(v)
    a = np.sign(a) * np.floor(np.abs(a) + 0.5)  # round halves away from zero
    return _num2str(interp, [a], 1)


@builtin("mat2str")
def _mat2str(interp, args, nargout):
    nargs(args, 1, 3, "mat2str")
    v = args[0]
    digits = 15
    with_class = False
    for opt in args[1:]:
        if is_text(opt):
            if text(opt).lower() != "class":
                raise MatlabError("mat2str: the option must be 'class'.")
            with_class = True
        else:
            digits = to_int(opt, "precision")
    if isinstance(v, MString):
        return char('"' + str(v).replace('"', '""') + '"')
    if is_char(v):
        rows = ["'" + "".join(r).replace("'", "''") + "'" for r in v] if v.shape[0] != 1 else \
            ["'" + to_str(v).replace("'", "''") + "'"]
        return char(rows[0] if len(rows) == 1 else "[" + ";".join(rows) + "]")
    if not isinstance(v, np.ndarray):
        raise MatlabError("mat2str: the input must be a numeric, logical or char array.")
    if v.ndim > 2:
        raise MatlabError("mat2str: the input must be a 2-D array.")
    a = v

    def f(x):
        if isinstance(x, (bool, np.bool_)):
            return "true" if x else "false"
        if isinstance(x, (complex, np.complexfloating)):
            return _num2str_scalar(x, None, digits)
        if isinstance(x, np.integer):
            return str(int(x))
        x = float(x)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Inf" if x > 0 else "-Inf"
        return f"{x + 0.0:.{digits}g}"

    if a.size == 0:
        body = f"zeros({a.shape[0]},{a.shape[1]})"
    elif a.size == 1:
        body = f(a.flat[0])
    else:
        body = "[" + ";".join(" ".join(f(x) for x in row) for row in a) + "]"
    cls = "logical" if a.dtype == bool else ("single" if a.dtype == np.float32 else a.dtype.name)
    if cls == "float64" or cls.startswith("complex"):
        cls = "single" if a.dtype == np.complex64 else "double"
    if with_class and cls != "logical":
        body = f"{cls}({body})"
    return char(body)


@builtin("str2double")
def _str2double(interp, args, nargout):
    v = args[0]
    if isinstance(v, (Cell, StringArray)):
        vals = [float("nan") if x is None else _parse_double(char(x) if isinstance(x, str) else x)
                for x in v.data.flatten(order="F")]
        cplx = any(isinstance(x, complex) for x in vals)
        out = np.array(vals, dtype=complex if cplx else float)
        return out.reshape(v.shape, order="F")
    if isinstance(v, np.ndarray) and not is_char(v):
        return float("nan")
    return _parse_double(v)


def _parse_double(v) -> float:
    try:
        s = to_str(v).strip().replace(",", "")
    except MatlabError:
        return float("nan")
    s = s.replace("i", "j") if s.endswith(("i", "j")) else s
    try:
        x = complex(s) if s.endswith("j") else float(s)
    except ValueError:
        low = s.lower()
        if low in ("inf", "+inf"):
            return float("inf")
        if low == "-inf":
            return float("-inf")
        return float("nan")
    return x  # type: ignore[return-value]


@builtin("str2num")
def _str2num(interp, args, nargout):
    v = args[0]
    if is_char(v) and v.ndim == 2 and v.shape[0] > 1:
        s = ";".join("".join(r) for r in v)
    else:
        s = text(v)
    try:
        val = interp.eval_expression("[" + s + "]")
        ok = True
    except MatlabError:
        val, ok = np.zeros((0, 0)), False
    return (val, bool_scalar(ok)) if nargout > 1 else val


@builtin("string")
def _string(interp, args, nargout):
    if not args:
        return MString("")
    v = args[0]
    if isinstance(v, (MString, StringArray)):
        return v
    if is_char(v) and v.ndim == 2 and v.shape[0] <= 1:
        return MString(to_str(v))
    return string_result(string_data(v))


@builtin("strings")
def _strings(interp, args, nargout):
    from ._util import dims_from_args
    shape, _ = dims_from_args(args, "strings") if args else ((1, 1), None)
    data = np.empty(shape, dtype=object)
    data[...] = ""
    return string_result(data) if data.size != 1 else MString("")


@builtin("strlength")
def _strlength(interp, args, nargout):
    v = args[0]
    if isinstance(v, (StringArray, Cell)):
        d = string_data(v)
        out = np.array([np.nan if x is None else float(len(x)) for x in d.flatten(order="F")])
        return out.reshape(d.shape, order="F")
    return float(len(text(v)))


@builtin("ismissing")
def _ismissing(interp, args, nargout):
    v = args[0]
    if isinstance(v, StringArray):
        return np.vectorize(lambda x: x is None, otypes=[bool])(v.data)
    if isinstance(v, MString):
        return np.array([[False]])
    if isinstance(v, np.ndarray) and v.dtype.kind in "fc":
        return np.isnan(v)
    if isinstance(v, np.ndarray):
        return np.zeros(v.shape, dtype=bool)
    if isinstance(v, Cell):
        return np.vectorize(lambda x: is_char(x) and x.size == 0, otypes=[bool])(v.data)
    raise MatlabError("ismissing: unsupported input type.")


@builtin("newline")
def _newline(interp, args, nargout):
    return char("\n")


@builtin("blanks")
def _blanks(interp, args, nargout):
    return char(" " * to_int(args[0], "n"))


# -- comparison and search -----------------------------------------------------------------

def _str_or_none(v):
    try:
        return to_str(v)
    except MatlabError:
        return None


def _cmp_builtin(fn):
    def f(interp, args, nargout):
        nargs(args, 2, 3, "strcmp")
        a, b = _as_cell_if_strings(args[0]), _as_cell_if_strings(args[1])
        extra = args[2:] if len(args) > 2 else []
        if isinstance(a, Cell) or isinstance(b, Cell):
            if isinstance(a, Cell) and isinstance(b, Cell):
                if a.data.size == 1:
                    a, b = b, a
                if b.data.size == 1:
                    s = _str_or_none(b.data.flat[0])
                    return np.vectorize(lambda x: s is not None and fn(_str_or_none(x), s, extra),
                                        otypes=[bool])(a.data) if a.data.size else np.zeros(a.shape, bool)
                if a.shape != b.shape:
                    raise MatlabError("Inputs must be the same size or either one can be a scalar.")
                return np.array([[fn(_str_or_none(x), _str_or_none(y), extra)
                                  for x, y in zip(ra, rb)] for ra, rb in zip(a.data, b.data)], dtype=bool)
            cell, other = (a, b) if isinstance(a, Cell) else (b, a)
            s = _str_or_none(other)
            if cell.data.size == 0:
                return np.zeros(cell.shape, bool)
            return np.vectorize(lambda x: s is not None and fn(_str_or_none(x), s, extra), otypes=[bool])(cell.data)
        sa, sb = _str_or_none(a), _str_or_none(b)
        if is_char(a) and a.shape[0] > 1 or is_char(b) and b.shape[0] > 1:
            return bool_scalar(isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                               and a.shape == b.shape and bool(np.all(a == b)))
        return bool_scalar(sa is not None and sb is not None and fn(sa, sb, extra))
    return f


def _n(extra):
    return to_int(extra[0], "n") if extra else None


builtin("strcmp")(_cmp_builtin(lambda a, b, e: a is not None and a == b))
builtin("strcmpi")(_cmp_builtin(lambda a, b, e: a is not None and a.lower() == b.lower()))
builtin("strncmp")(_cmp_builtin(lambda a, b, e: a is not None and len(a) >= _n(e) and len(b) >= _n(e)
                                and a[:_n(e)] == b[:_n(e)]))
builtin("strncmpi")(_cmp_builtin(lambda a, b, e: a is not None and len(a) >= _n(e) and len(b) >= _n(e)
                                 and a[:_n(e)].lower() == b[:_n(e)].lower()))


def _map_text(fn, v):
    """Apply a str->str function to char, string or each element of a cell."""
    if isinstance(v, StringArray):
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            x = v.data[idx]
            data[idx] = None if x is None else fn(x)
        return string_result(data)
    if isinstance(v, Cell):
        data = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            data[idx] = _map_text(fn, v.data[idx])
        return Cell(data)
    if isinstance(v, MString):
        return MString(fn(str(v)))
    if is_char(v):
        if v.shape[0] > 1:
            rows = [fn("".join(r)) for r in v]
            w = max(len(r) for r in rows)
            return np.array([list(r.ljust(w)) for r in rows], dtype="<U1")
        return char(fn(to_str(v)))
    raise MatlabError("Expected text input.")


def _case_map(fn):
    """Per-character case mapping that never changes the text length (e.g. 'ß' stays)."""
    def conv(s: str) -> str:
        out = []
        for c in s:
            m = fn(c)
            out.append(m if len(m) == 1 else c)
        return "".join(out)
    return conv


def _case_builtin(fn):
    conv = _case_map(fn)

    def f(interp, args, nargout):
        nargs(args, 1, 1, "upper")
        v = args[0]
        if isinstance(v, (MString, StringArray, Cell)) or is_char(v):
            return _map_text(conv, v)
        return v  # numbers and other values come back unchanged

    return f


def _trim_builtin(leading: bool):
    def one(s: str) -> str:
        return s.strip(_BLANKS) if leading else s.rstrip(_BLANKS)

    def f(interp, args, nargout):
        nargs(args, 1, 1, "strtrim" if leading else "deblank")
        v = args[0]
        if is_char(v) and v.ndim == 2 and v.shape[0] > 1:
            # a char matrix loses only the blank columns common to all rows
            return _rows_to_char(_trim_rows(["".join(r) for r in v], leading))
        if is_char(v) and v.size == 0:
            return v
        if isinstance(v, np.ndarray) and not is_char(v):
            if leading or v.size:
                raise MatlabError("Input should be a string, character array, or a cell array of character arrays.")
            return v
        return _map_text(one, v)

    return f


builtin("upper", "toupper")(_case_builtin(str.upper))
builtin("lower", "tolower")(_case_builtin(str.lower))
builtin("strtrim")(_trim_builtin(True))
builtin("deblank")(_trim_builtin(False))
builtin("reverse")(lambda interp, args, nargout: _map_text(lambda s: s[::-1], args[0]))


def _find_all(s: str, pat: str) -> list[int]:
    """Start offsets (0-based) of every occurrence of pat in s, overlapping ones included."""
    out = []
    if not pat:
        return out
    start = s.find(pat)
    while start != -1:
        out.append(start)
        start = s.find(pat, start + 1)
    return out


def _replace_all(s: str, old: str, new: str) -> str:
    """strrep: every occurrence is replaced, also the ones that overlap an earlier one."""
    hits = _find_all(s, old)
    if not hits:
        return s
    out, pos = [], 0
    for h in hits:
        if h > pos:
            out.append(s[pos:h])
        out.append(new)
        pos = h + len(old)
    out.append(s[pos:])
    return "".join(out)


@builtin("strrep")
def _strrep(interp, args, nargout):
    nargs(args, 3, 3, "strrep")
    old, new = text(args[1]), text(args[2])
    return _map_text(lambda s: _replace_all(s, old, new), args[0])


@builtin("strfind")
def _strfind(interp, args, nargout):
    nargs(args, 2, 4, "strfind")
    pat = text(args[1])
    overlaps = True
    if len(args) == 4 and text(args[2]).lower() == "overlaps":
        overlaps = bool(num(args[3]).flat[0])

    def find(s):
        hits = _find_all(s, pat)
        if not overlaps:
            kept, last = [], -1
            for h in hits:
                if h >= last:
                    kept.append(h)
                    last = h + len(pat)
            hits = kept
        return np.array(hits, dtype=float).reshape(1, -1) + 1.0 if hits else np.zeros((0, 0))

    v = args[0]
    if isinstance(v, Cell):
        data = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            data[idx] = find(to_str(v.data[idx]))
        return Cell(data)
    return find(text(v))


def _text_pred(fn):
    def f(interp, args, nargout):
        rest = list(args)
        ignore_case = False
        if len(rest) >= 4 and is_text(rest[-2]) and text(rest[-2]).lower() == "ignorecase":
            ignore_case = bool(num(rest[-1]).flat[0])
            rest = rest[:-2]
        pat_val = _as_cell_if_strings(rest[1])
        pats = [to_str(x) for x in pat_val.values_f()] if isinstance(pat_val, Cell) else [text(pat_val)]

        def test(s):
            if ignore_case:
                return any(fn(s.lower(), p.lower()) for p in pats)
            return any(fn(s, p) for p in pats)

        v = _as_cell_if_strings(rest[0])
        if isinstance(v, Cell):
            return np.vectorize(lambda x: test(to_str(x)), otypes=[bool])(v.data) if v.data.size else \
                np.zeros(v.shape, bool)
        return bool_scalar(test(text(v)))
    return f


builtin("contains")(_text_pred(lambda s, p: p in s))
builtin("startsWith")(_text_pred(lambda s, p: s.startswith(p)))
builtin("endsWith")(_text_pred(lambda s, p: s.endswith(p)))


@builtin("strsplit")
def _strsplit(interp, args, nargout):
    nargs(args, 1, None, "strsplit")
    s = text(args[0])
    want_strings = isinstance(args[0], (MString, StringArray))
    rest = list(args[1:])
    delim_arg = None
    if rest and (len(rest) % 2 == 1 or not (is_text(rest[0]) and text(rest[0]).lower() in
                                             ("collapsedelimiters", "delimitertype"))):
        delim_arg = rest.pop(0)
    collapse, regex = True, False
    for k in range(0, len(rest) - 1, 2):
        name = text(rest[k]).lower()
        if name == "collapsedelimiters":
            collapse = bool(np.all(num(rest[k + 1]) != 0))
        elif name == "delimitertype":
            regex = text(rest[k + 1]).lower() == "regularexpression"
        else:
            raise MatlabError(f"Unknown strsplit option '{text(rest[k])}'.")
    if delim_arg is None:
        pattern = r"\s"
    else:
        d = _as_cell_if_strings(delim_arg)
        delims = [to_str(x) for x in d.values_f()] if isinstance(d, Cell) else [text(d)]
        if regex:
            pattern = "|".join(f"(?:{x})" for x in delims)
        else:
            delims = sorted((process_escapes(x) for x in delims), key=len, reverse=True)
            pattern = "|".join(re.escape(x) for x in delims)
    if collapse:
        pattern = f"(?:{pattern})+"
    rx = re.compile(pattern)
    parts, matches, last = [], [], 0
    for m in rx.finditer(s):
        if m.end() == m.start():
            continue
        parts.append(s[last:m.start()])
        matches.append(m.group(0))
        last = m.end()
    parts.append(s[last:])

    def wrap(items):
        if want_strings:
            data = np.empty((1, len(items)), dtype=object)
            for k, x in enumerate(items):
                data[0, k] = x
            return StringArray(data) if len(items) != 1 else string_result(data)
        return Cell.from_list([char(p) for p in items])

    if nargout > 1:
        return wrap(parts), wrap(matches)
    return wrap(parts)


@builtin("strtok")
def _strtok(interp, args, nargout):
    nargs(args, 1, 2, "strtok")
    delims = process_escapes(text(args[1])) if len(args) > 1 else " \t\n\v\f\r\0"

    def tok(s: str):
        i = 0
        while i < len(s) and s[i] in delims:
            i += 1
        j = i
        while j < len(s) and s[j] not in delims:
            j += 1
        return s[i:j], s[j:]

    v = args[0]
    if isinstance(v, (Cell, StringArray)):
        src = _as_cell_if_strings(v)
        toks = np.empty(src.shape, dtype=object)
        rems = np.empty(src.shape, dtype=object)
        for idx in np.ndindex(src.shape):
            toks[idx], rems[idx] = tok(to_str(src.data[idx]))
        if isinstance(v, StringArray):
            return (string_result(toks), string_result(rems)) if nargout > 1 else string_result(toks)
        tc = Cell(np.vectorize(char, otypes=[object])(toks)) if toks.size else Cell(toks)
        rc = Cell(np.vectorize(char, otypes=[object])(rems)) if rems.size else Cell(rems)
        return (tc, rc) if nargout > 1 else tc
    t, r = tok(text(v))
    if isinstance(v, MString):
        return (MString(t), MString(r)) if nargout > 1 else MString(t)
    return (char(t), char(r)) if nargout > 1 else char(t)


@builtin("splitlines")
def _splitlines(interp, args, nargout):
    nargs(args, 1, 1, "splitlines")
    parts = re.split(r"\r\n|\n|\r", text(args[0]))
    if isinstance(args[0], MString):
        data = np.empty((len(parts), 1), dtype=object)
        for k, x in enumerate(parts):
            data[k, 0] = x
        return string_result(data)
    return Cell.from_list([char(p) for p in parts], column=True)


@builtin("missing")
def _missing(interp, args, nargout):
    data = np.empty((1, 1), dtype=object)
    data[0, 0] = None
    return StringArray(data)


@builtin("strjoin")
def _strjoin(interp, args, nargout):
    src = _as_cell_if_strings(args[0])
    if not isinstance(src, Cell):
        raise MatlabError("strjoin: the first input must be a cell array of character vectors.")
    items = [to_str(x) for x in src.values_f()]
    dval = _as_cell_if_strings(args[1]) if len(args) > 1 else None
    if isinstance(dval, Cell):
        delims = [to_str(x) for x in dval.values_f()]
        if len(delims) != max(len(items) - 1, 0):
            raise MatlabError("strjoin: the number of delimiters must be one less than the number of pieces.")
        joined = "".join(x + d for x, d in zip(items, delims + [""]))
    else:
        delim = process_escapes(text(dval)) if dval is not None else " "
        joined = delim.join(items)
    return MString(joined) if isinstance(args[0], (StringArray, MString)) else char(joined)


@builtin("strcat")
def _strcat(interp, args, nargout):
    if any(isinstance(a, Cell) for a in args):
        n = max(a.data.size for a in args if isinstance(a, Cell))
        shape = next(a.shape for a in args if isinstance(a, Cell) and a.data.size == n)
        out = []
        for k in range(n):
            pieces = []
            for a in args:
                if isinstance(a, Cell):
                    pieces.append(to_str(a.data.flatten(order="F")[k if a.data.size > 1 else 0]))
                else:
                    pieces.append(text(a))
            out.append(char("".join(pieces)))
        c = Cell.from_list(out)
        return Cell(c.data.reshape(shape, order="F"))
    if any(isinstance(a, MString) for a in args):
        return MString("".join(text(a) for a in args))
    if any(is_char(a) and a.ndim == 2 and a.shape[0] > 1 for a in args):
        # char matrices are joined row by row; single rows are repeated
        nrows = max(a.shape[0] for a in args if is_char(a))
        blocks = []
        for a in args:
            rows = ["".join(r) for r in a] if is_char(a) and a.size else [text(a)]
            if len(rows) == 1:
                rows = rows * nrows
            if len(rows) != nrows:
                raise MatlabError("strcat: the inputs must have the same number of rows.")
            blocks.append(_trim_rows(rows, leading=False))
        return _rows_to_char(["".join(parts) for parts in zip(*blocks)])
    return char("".join(text(a).rstrip(" \t\n\v\f\r\0") if is_char(a) else text(a) for a in args))


@builtin("strvcat")
def _strvcat(interp, args, nargout):
    from .core import _char_rows
    return _char_rows([text(a) for a in args if text(a)])


@builtin("isspace")
def _isspace(interp, args, nargout):
    v = args[0]
    if not is_char(v):
        return np.zeros(np.shape(v) if isinstance(v, np.ndarray) else (1, 1), dtype=bool)
    return np.vectorize(lambda c: c.isspace(), otypes=[bool])(v) if v.size else np.zeros(v.shape, bool)


@builtin("isletter")
def _isletter(interp, args, nargout):
    v = args[0]
    if not is_char(v) or v.size == 0:
        return np.zeros(np.shape(v), dtype=bool)
    return np.vectorize(lambda c: c.isalpha(), otypes=[bool])(v)


@builtin("isstrprop")
def _isstrprop(interp, args, nargout):
    v, prop = args[0], text(args[1])
    tests = {
        "alpha": str.isalpha, "digit": str.isdigit, "alphanum": str.isalnum, "wspace": str.isspace,
        "upper": str.isupper, "lower": str.islower, "punct": lambda c: (not c.isalnum()) and c.isprintable()
        and not c.isspace(), "cntrl": lambda c: not c.isprintable(),
        "xdigit": lambda c: c in "0123456789abcdefABCDEF",
        "graphic": lambda c: c.isprintable() and not c.isspace(),
        "print": lambda c: c.isprintable(),
        "ascii": lambda c: ord(c) < 128,
    }
    if isinstance(v, Cell):
        data = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            data[idx] = _isstrprop(interp, [v.data[idx], args[1]], 1)
        return Cell(data)
    if isinstance(v, MString):
        v = char(str(v))
    if isinstance(v, np.ndarray) and not is_char(v):
        from ..values import codes_to_char
        v = codes_to_char(v)
    if prop not in tests:
        raise MatlabError(f"Unknown property '{prop}'.")
    if v.size == 0:
        return np.zeros(v.shape, bool)
    return np.vectorize(tests[prop], otypes=[bool])(v)


# -- regular expressions -------------------------------------------------------------------

_POSIX_CLASSES = {
    "alpha": "a-zA-Z", "digit": "0-9", "alnum": "a-zA-Z0-9", "upper": "A-Z", "lower": "a-z",
    "space": r"\s", "xdigit": "0-9A-Fa-f", "word": r"\w", "blank": r" \t", "cntrl": r"\x00-\x1f\x7f",
    "punct": r"!-/:-@\[-`{-~", "graph": r"!-~", "print": r" -~",
}

_OUTPUT_KEYS = ("start", "end", "tokenextents", "match", "tokens", "names", "split")
_REGEX_SWITCHES = {
    "once", "ignorecase", "matchcase", "emptymatch", "noemptymatch", "dotall", "dotexceptnewline",
    "stringanchors", "lineanchors", "literalspacing", "freespacing", "warnings", "forcecelloutput",
    "preservecase",
}


def _translate_regex(p: str, multiline: bool = False) -> str:
    """Rewrite a MATLAB regular expression in Python's dialect."""
    out: list[str] = []
    i, n = 0, len(p)
    in_class = False
    while i < n:
        c = p[i]
        if c == "\\" and i + 1 < n:
            d = p[i + 1]
            if not in_class and d == "<":
                out.append(r"\b(?=\w)")
                i += 2
                continue
            if not in_class and d == ">":
                out.append(r"\b(?<=\w)")
                i += 2
                continue
            m = re.match(r"\\k<([A-Za-z]\w*)>", p[i:]) if not in_class else None
            if m:
                out.append(f"(?P={m.group(1)})")
                i += m.end()
                continue
            m = re.match(r"\\o\{([0-7]+)\}|\\o([0-7]{1,3})|\\x\{([0-9A-Fa-f]+)\}", p[i:])
            if m:
                code = int(m.group(1) or m.group(2), 8) if m.group(3) is None else int(m.group(3), 16)
                out.append(re.escape(chr(code)))
                i += m.end()
                continue
            out.append(c + d)
            i += 2
            continue
        if in_class:
            m = re.match(r"\[:([a-z]+):\]", p[i:]) if c == "[" else None
            if m and m.group(1) in _POSIX_CLASSES:
                out.append(_POSIX_CLASSES[m.group(1)])
                i += m.end()
                continue
            if c == "[":
                out.append("\\[")
            else:
                if c == "]":
                    in_class = False
                out.append(c)
            i += 1
            continue
        if c == "[":
            in_class = True
            out.append(c)
            i += 1
            if p[i:i + 1] == "^":
                out.append("^")
                i += 1
            if p[i:i + 1] == "]":
                out.append("\\]")
                i += 1
            continue
        if c == "(" and p.startswith("(?<", i) and p[i + 3:i + 4] not in ("=", "!"):
            out.append("(?P<")
            i += 3
            continue
        if c == "$" and not multiline:
            out.append(r"\Z")  # '$' is the end of the text, also when it ends with a newline
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _regex_options(opts: list, name: str) -> tuple[list[str], set[str]]:
    """Split the trailing options into the selected outputs (in order) and the switches."""
    selected: list[str] = []
    switches: set[str] = set()
    for o in opts:
        if not is_text(o):
            raise MatlabError(f"{name}: options must be text.")
        key = text(o).lower()
        if key in _OUTPUT_KEYS:
            selected.append(key)
        elif key in _REGEX_SWITCHES:
            switches.add(key)
        else:
            raise MatlabError(f"{name}: unknown option '{text(o)}'.")
    return selected, switches


def _compile_regex(pattern: str, switches: set[str], name: str):
    """Compiled pattern, or None for an empty pattern (which never matches)."""
    if pattern == "":
        return None
    flags = 0
    if "ignorecase" in switches and "matchcase" not in switches:
        flags |= re.IGNORECASE
    if "dotexceptnewline" not in switches:
        flags |= re.DOTALL
    if "lineanchors" in switches:
        flags |= re.MULTILINE
    if "freespacing" in switches:
        flags |= re.VERBOSE
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return re.compile(_translate_regex(pattern, "lineanchors" in switches), flags)
    except (re.error, RecursionError, OverflowError) as e:
        raise MatlabError(f"{name}: invalid regular expression '{pattern}' ({e}).") from None


def _regex_matches(rx, s: str, switches: set[str]) -> list:
    out = []
    if rx is None:
        return out
    keep_empty = "emptymatch" in switches
    pos = 0
    while pos <= len(s):
        m = rx.search(s, pos)
        if m is None:
            break
        if m.end() == m.start():
            # an empty match is dropped (unless asked for) and the scan moves on one character
            pos = m.end() + 1
            if not keep_empty:
                continue
        else:
            pos = m.end()
        out.append(m)
        if "once" in switches:
            break
    return out


def _row(values) -> np.ndarray:
    return np.array(values, dtype=float).reshape(1, -1) if values else np.zeros((1, 0))


def _regexp_one(rx, s: str, switches: set[str]) -> dict:
    """All seven outputs of regexp for one piece of text."""
    from ..values import Struct
    matches = _regex_matches(rx, s, switches)
    ngroups = rx.groups if rx is not None else 0
    group_names = list(rx.groupindex) if rx is not None else []

    def tokens_of(m):
        if not ngroups:
            return Cell.from_list([char(m.group(0))])
        return Cell.from_list([char(g or "") for g in m.groups()])

    def extents_of(m):
        if not ngroups:
            return np.array([[m.start() + 1.0, float(m.end())]])
        rows = []
        for g in range(1, ngroups + 1):
            a, b = m.span(g)
            rows.append([a + 1.0, float(b)] if a >= 0 else [m.end() + 1.0, float(m.end())])
        return np.array(rows, dtype=float)

    def names_of(ms):
        if not group_names:
            return Struct()
        if not ms:
            return Struct(np.empty((0, 0), dtype=object), group_names)
        sdata = np.empty((1, len(ms)), dtype=object)
        for k, m in enumerate(ms):
            sdata[0, k] = {name: char(m.group(name) or "") for name in group_names}
        return Struct(sdata, group_names)

    pieces, last = [], 0
    for m in matches:
        pieces.append(char(s[last:m.start()]))
        last = m.end()
    pieces.append(char(s[last:]))
    res = {"split": Cell.from_list(pieces), "names": names_of(matches)}
    if "once" in switches:
        if matches:
            m = matches[0]
            res.update({"start": np.array([[m.start() + 1.0]]), "end": np.array([[float(m.end())]]),
                        "tokenextents": extents_of(m), "match": char(m.group(0)), "tokens": tokens_of(m)})
        else:
            res.update({"start": np.zeros((0, 0)), "end": np.zeros((0, 0)), "tokenextents": np.zeros((0, 0)),
                        "match": char(""), "tokens": Cell(np.empty((0, 0), dtype=object))})
    else:
        res.update({
            "start": _row([m.start() + 1.0 for m in matches]),
            "end": _row([float(m.end()) for m in matches]),
            "tokenextents": Cell.from_list([extents_of(m) for m in matches]),
            "match": Cell.from_list([char(m.group(0)) for m in matches]),
            "tokens": Cell.from_list([tokens_of(m) for m in matches]),
        })
    return res


def _text_items(v, name: str):
    """(list of str, shape or None): the texts of a char/string/cell argument."""
    v = _as_cell_if_strings(v)
    if isinstance(v, Cell):
        return [to_str(x) for x in v.data.flatten(order="F")], v.shape
    return [text(v)], None


@builtin("regexp")
def _regexp(interp, args, nargout):
    nargs(args, 2, None, "regexp")
    selected, switches = _regex_options(list(args[2:]), "regexp")
    strs, s_shape = _text_items(args[0], "regexp")
    pats, p_shape = _text_items(args[1], "regexp")
    order = selected or list(_OUTPUT_KEYS)
    n = max(nargout, 1)
    if n > len(order):
        raise MatlabError("regexp: too many output arguments.")
    order = order[:n]
    compiled = [_compile_regex(p, switches, "regexp") for p in pats]
    if s_shape is None and p_shape is None and "forcecelloutput" not in switches:
        res = _regexp_one(compiled[0], strs[0], switches)
        out = [res[k] for k in order]
    else:
        if s_shape is not None and p_shape is not None and len(strs) != len(pats) \
                and len(strs) != 1 and len(pats) != 1:
            raise MatlabError("regexp: the cell arrays of text and of expressions must have the same size.")
        shape = s_shape if s_shape is not None and (p_shape is None or len(strs) >= len(pats)) else p_shape
        if shape is None:
            shape = (1, 1)
        count = max(len(strs), len(pats)) if strs and pats else 0
        cols = [np.empty(count, dtype=object) for _ in order]
        for k in range(count):
            res = _regexp_one(compiled[k if len(pats) > 1 else 0], strs[k if len(strs) > 1 else 0], switches)
            for col, key in zip(cols, order):
                col[k] = res[key]
        out = [Cell(col.reshape(shape, order="F")) for col in cols]
    return tuple(out) if n > 1 else out[0]


@builtin("regexpi")
def _regexpi(interp, args, nargout):
    opts = [text(o).lower() for o in args[2:] if is_text(o)]
    extra = [] if "matchcase" in opts else [char("ignorecase")]
    return _regexp(interp, [*args, *extra], nargout)


_REPL_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a", "b": "\b"}


def _parse_replacement(r: str) -> list[tuple[str, object]]:
    """Split a regexprep replacement into literal text, token references and ${...} expressions."""
    parts: list[tuple[str, object]] = []
    lit: list[str] = []

    def flush():
        if lit:
            parts.append(("lit", "".join(lit)))
            lit.clear()

    i, n = 0, len(r)
    while i < n:
        c = r[i]
        if c == "\\" and i + 1 < n:
            d = r[i + 1]
            m = re.match(r"\\o\{([0-7]+)\}|\\o([0-7]{1,3})|\\x\{([0-9A-Fa-f]+)\}|\\x([0-9A-Fa-f]{1,2})", r[i:])
            if m:
                octal = m.group(1) or m.group(2)
                lit.append(chr(int(octal, 8) if octal else int(m.group(3) or m.group(4), 16)))
                i += m.end()
                continue
            lit.append(_REPL_ESCAPES.get(d, d))
            i += 2
            continue
        if c == "$" and i + 1 < n:
            m = re.match(r"\$(\d+)", r[i:])
            if m:
                flush()
                parts.append(("group", m.group(1)))
                i += m.end()
                continue
            m = re.match(r"\$<([A-Za-z]\w*)>", r[i:])
            if m:
                flush()
                parts.append(("name", m.group(1)))
                i += m.end()
                continue
            if r[i + 1] == "{":
                depth, j = 0, i + 1
                while j < n:
                    if r[j] == "{":
                        depth += 1
                    elif r[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j < n:
                    flush()
                    parts.append(("expr", r[i + 2:j]))
                    i = j + 1
                    continue
        lit.append(c)
        i += 1
    flush()
    return parts


def _group_text(m, digits: str) -> tuple[str, str]:
    """Text of token $digits and the digits that are not part of the token number."""
    ngroups = m.re.groups
    k = len(digits)
    while k > 1 and int(digits[:k]) > ngroups:
        k -= 1
    idx = int(digits[:k])
    if idx > ngroups:
        return "", digits[k:]
    return m.group(idx) or "", digits[k:]


def _render_replacement(interp, parts, m) -> str:
    out = []
    for kind, val in parts:
        if kind == "lit":
            out.append(val)
        elif kind == "group":
            out.extend(_group_text(m, val))
        elif kind == "name":
            out.append((m.groupdict().get(val) or "") if val in m.re.groupindex else f"$<{val}>")
        else:
            def quote(t: str) -> str:
                return "'" + t.replace("'", "''") + "'"

            code = re.sub(r"\$<([A-Za-z]\w*)>", lambda g: quote(m.groupdict().get(g.group(1)) or ""), val)
            code = re.sub(r"\$(\d+)", lambda g: quote(_group_text(m, g.group(1))[0]) + _group_text(m, g.group(1))[1],
                          code)
            res = interp.eval_expression(code)
            if is_text(res):
                out.append(text(res))
            elif isinstance(res, np.ndarray):
                out.append(to_str(_num2str(interp, [res], 1)))
            else:
                raise MatlabError("regexprep: a ${...} expression must return text.")
    return "".join(out)


def _match_case(src: str, rep: str) -> str:
    if src and src.isupper():
        return rep.upper()
    if src and src.islower():
        return rep.lower()
    if src[:1].isupper() and src[1:].islower():
        return rep[:1].upper() + rep[1:].lower()
    return rep


@builtin("regexprep")
def _regexprep(interp, args, nargout):
    nargs(args, 3, None, "regexprep")
    selected, switches = _regex_options(list(args[3:]), "regexprep")
    if selected:
        raise MatlabError(f"regexprep: unknown option '{selected[0]}'.")
    pats, _ = _text_items(args[1], "regexprep")
    reps, rep_shape = _text_items(args[2], "regexprep")
    if rep_shape is not None and len(reps) != 1 and len(reps) != len(pats):
        raise MatlabError("regexprep: the number of replacements must match the number of expressions.")
    if len(reps) == 1:
        reps = reps * len(pats)
    steps = [(_compile_regex(p, switches, "regexprep"), _parse_replacement(r)) for p, r in zip(pats, reps)]

    def apply(s: str) -> str:
        for rx, parts in steps:
            matches = _regex_matches(rx, s, switches)
            if not matches:
                continue
            out, pos = [], 0
            for m in matches:
                out.append(s[pos:m.start()])
                new = _render_replacement(interp, parts, m)
                out.append(_match_case(m.group(0), new) if "preservecase" in switches else new)
                pos = m.end()
            out.append(s[pos:])
            s = "".join(out)
        return s

    return _map_text(apply, args[0])


# -- numbers in other bases ----------------------------------------------------------------

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _base_symbols(base, name: str) -> str:
    if is_text(base):
        symbols = text(base)
        if len(symbols) < 2 or len(set(symbols)) != len(symbols) or " " in symbols:
            raise MatlabError(f"{name}: the symbols must be at least two different non-blank characters.")
        return symbols
    b = to_int(base, "base")
    if not 2 <= b <= 36:
        raise MatlabError(f"{name}: the base must be an integer between 2 and 36.")
    return _DIGITS[:b]


def _to_base(value, symbols: str, name: str) -> str:
    v = float(value) if not isinstance(value, (int, np.integer)) else int(value)
    if isinstance(v, float):
        if not math.isfinite(v) or v != math.floor(v):
            raise MatlabError(f"{name}: the input must be a non-negative integer.")
        v = int(v)
    if v < 0:
        raise MatlabError(f"{name}: the input must be a non-negative integer.")
    base = len(symbols)
    out = []
    while True:
        v, r = divmod(v, base)
        out.append(symbols[r])
        if v == 0:
            break
    return "".join(reversed(out))


def _dec2base(value, symbols: str, min_len: int, name: str):
    if isinstance(value, Cell):
        items = [x.flat[0] for x in value.values_f()]
    else:
        a = value if isinstance(value, np.ndarray) and value.dtype.kind in "iu" else num(value)
        if a.dtype.kind == "c":
            raise MatlabError(f"{name}: the input must be real.")
        items = list(a.flatten(order="F"))
    if not items:
        return char("")
    rows = [_to_base(x, symbols, name) for x in items]
    width = max(max(len(r) for r in rows), min_len)
    return _rows_to_char([r.rjust(width, symbols[0]) for r in rows])


def _base2dec(value, symbols: str, name: str):
    value = _as_cell_if_strings(value)
    if isinstance(value, Cell):
        rows = [to_str(x) for x in value.values_f()]
    elif isinstance(value, MString):
        rows = [str(value)]
    elif is_char(value):
        rows = ["".join(r) for r in value] if value.size else []
    else:
        raise MatlabError(f"{name}: the input must be text.")
    base = len(symbols)
    lookup = {c: k for k, c in enumerate(symbols)}
    if symbols == _DIGITS[:base]:
        lookup.update({c.lower(): k for c, k in lookup.items()})
    out = []
    for r in rows:
        r = r.replace(" ", "")
        total = 0
        ok = bool(r)
        for c in r:
            if c not in lookup:
                ok = False
                break
            total = total * base + lookup[c]
        out.append(float(total) if ok else float("nan"))
    if not out:
        return np.zeros((0, 0))
    return np.array(out, dtype=float).reshape(-1, 1)


@builtin("dec2base")
def _dec2base_fn(interp, args, nargout):
    nargs(args, 2, 3, "dec2base")
    n = to_int(args[2], "length") if len(args) > 2 else 0
    return _dec2base(args[0], _base_symbols(args[1], "dec2base"), n, "dec2base")


@builtin("base2dec")
def _base2dec_fn(interp, args, nargout):
    nargs(args, 2, 2, "base2dec")
    return _base2dec(args[0], _base_symbols(args[1], "base2dec"), "base2dec")


def _dec2(base: int, name: str):
    def f(interp, args, nargout):
        nargs(args, 1, 2, name)
        n = to_int(args[1], "length") if len(args) > 1 else 0
        return _dec2base(args[0], _DIGITS[:base], n, name)
    return f


def _2dec(base: int, name: str):
    def f(interp, args, nargout):
        nargs(args, 1, 1, name)
        return _base2dec(args[0], _DIGITS[:base], name)
    return f


builtin("dec2bin")(_dec2(2, "dec2bin"))
builtin("dec2hex")(_dec2(16, "dec2hex"))
builtin("bin2dec")(_2dec(2, "bin2dec"))
builtin("hex2dec")(_2dec(16, "hex2dec"))
