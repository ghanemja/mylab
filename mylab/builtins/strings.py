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

_SPEC = re.compile(r"%(?P<flags>[-+ 0#]*)(?P<width>\*|\d+)?(?:\.(?P<prec>\*|\d+))?(?P<conv>[diouxXfeEgGcs%])")

_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


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
        out.append(ch)
        i += 1
    return "".join(out)


def _flatten_args(args) -> list:
    items: list = []
    for a in args:
        if isinstance(a, MString):
            items.append(str(a))
        elif is_char(a):
            items.append(to_str(a))
        elif isinstance(a, StringArray):
            items.extend("" if x is None else x for x in a.data.flatten(order="F"))
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
    if conv in "diouxXc" and isinstance(value, str):
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
        return f"%{flags}{w}d" % int(value)
    if conv in "oxX":
        if float(value) != int(value):
            return f"%{flags}{w}{p or ''}e" % value
        return f"%{flags}{w}{conv}" % int(value)
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
    if a.size == 1:
        return char(_num2str_scalar(a.flat[0], fmt, digits))
    if a.size == 0:
        return char("")
    if fmt is not None:
        rows = [matlab_sprintf(fmt, [row.reshape(1, -1)]) for row in a]
    else:
        cells = [[_num2str_scalar(x, None, digits) for x in row] for row in a]
        all_int = a.dtype.kind in "biu" or bool(np.all(np.isfinite(a)) and np.all(np.real(a) == np.round(np.real(a))))
        width = max(len(c) for r in cells for c in r)
        sep = 2 if all_int else 4
        rows = [(" " * sep).join(c.rjust(width) for c in r) for r in cells]
    width = max(len(r) for r in rows)
    return np.array([list(r.rjust(width)) for r in rows], dtype="<U1").reshape(len(rows), width)


@builtin("int2str")
def _int2str(interp, args, nargout):
    a = fnum(args[0])
    a = np.sign(a) * np.floor(np.abs(a) + 0.5)  # round halves away from zero
    return _num2str(interp, [a], 1)


@builtin("mat2str")
def _mat2str(interp, args, nargout):
    v = args[0]
    digits = to_int(args[1], "precision") if len(args) > 1 else 15
    if is_char(v):
        return char("'" + to_str(v) + "'")
    a = num(v) if not (isinstance(v, np.ndarray) and v.dtype == bool) else v

    def f(x):
        if isinstance(x, (bool, np.bool_)):
            return "true" if x else "false"
        if isinstance(x, (complex, np.complexfloating)):
            return _num2str_scalar(x, None, digits)
        x = float(x)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Inf" if x > 0 else "-Inf"
        return f"{x:.{digits}g}"

    if a.size == 1:
        return char(f(a.flat[0]))
    body = ";".join(" ".join(f(x) for x in row) for row in a)
    return char(f"[{body}]")


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
    s = text(args[0])
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


builtin("upper")(lambda interp, args, nargout: _map_text(_case_map(str.upper), args[0]))
builtin("lower")(lambda interp, args, nargout: _map_text(_case_map(str.lower), args[0]))
builtin("strtrim")(lambda interp, args, nargout: _map_text(lambda s: s.strip(" \t\n\r\f\v\0"), args[0]))
builtin("deblank")(lambda interp, args, nargout: _map_text(lambda s: s.rstrip(" \t\n\r\f\v\0"), args[0]))
builtin("reverse")(lambda interp, args, nargout: _map_text(lambda s: s[::-1], args[0]))


@builtin("strrep")
def _strrep(interp, args, nargout):
    old, new = text(args[1]), text(args[2])
    return _map_text(lambda s: s.replace(old, new) if old else s, args[0])


@builtin("strfind")
def _strfind(interp, args, nargout):
    pat = text(args[1])

    def find(s):
        if not pat:
            return np.zeros((1, 0))
        out = []
        start = s.find(pat)
        while start != -1:
            out.append(start + 1.0)
            start = s.find(pat, start + 1)
        return np.array(out).reshape(1, -1) if out else np.zeros((1, 0))

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
    items = [to_str(x) for x in src.values_f()]
    delim = process_escapes(text(args[1])) if len(args) > 1 else " "
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
    return char("".join(text(a).rstrip(" \t\n\0") if is_char(a) else text(a) for a in args))


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
    }
    if prop not in tests:
        raise MatlabError(f"Unknown property '{prop}'.")
    if v.size == 0:
        return np.zeros(v.shape, bool)
    return np.vectorize(tests[prop], otypes=[bool])(v)


# -- regular expressions -------------------------------------------------------------------

def _translate_regex(p: str) -> str:
    # MATLAB named tokens (?<name>...) -> Python (?P<name>...)
    return re.sub(r"\(\?<([A-Za-z]\w*)>", r"(?P<\1>", p)


def _translate_replacement(r: str) -> str:
    r = re.sub(r"\$(\d+)", r"\\g<\1>", r)
    r = r.replace("$0", "\\g<0>")
    return process_escapes(r) if "\\" in r and not re.search(r"\\g<", r) else r


@builtin("regexprep")
def _regexprep(interp, args, nargout):
    pat = _translate_regex(text(args[1]))
    rep = _translate_replacement(text(args[2]))
    flags = 0
    count = 0
    for opt in args[3:]:
        o = text(opt).lower()
        if o == "ignorecase":
            flags |= re.IGNORECASE
        elif o == "once":
            count = 1
    rx = re.compile(pat, flags)
    return _map_text(lambda s: rx.sub(rep, s, count=count), args[0])


@builtin("regexpi")
def _regexpi(interp, args, nargout):
    return _regexp(interp, [*args, char("ignorecase")], nargout)


@builtin("regexp")
def _regexp(interp, args, nargout):
    s_val, pat = args[0], _translate_regex(text(args[1]))
    opts = [text(o).lower() for o in args[2:]]
    flags = re.IGNORECASE if "ignorecase" in opts else 0
    once = "once" in opts
    rx = re.compile(pat, flags)
    if isinstance(s_val, Cell):
        data = np.empty(s_val.shape, dtype=object)
        for idx in np.ndindex(s_val.shape):
            data[idx] = _regexp(interp, [s_val.data[idx], *args[1:]], 1)
        return Cell(data)
    s = text(s_val)
    matches = list(rx.finditer(s))
    if once:
        matches = matches[:1]

    def tokens_of(m):
        groups = m.groups()
        if not groups:
            return Cell.from_list([char(m.group(0))])
        return Cell.from_list([char(g or "") for g in groups])

    results = {
        "start": np.array([m.start() + 1.0 for m in matches]).reshape(1, -1) if matches else np.zeros((1, 0)),
        "end": np.array([float(m.end()) for m in matches]).reshape(1, -1) if matches else np.zeros((1, 0)),
        "match": Cell.from_list([char(m.group(0)) for m in matches]),
        "tokens": Cell.from_list([tokens_of(m) for m in matches]),
        "names": None,
        "split": None,
    }
    pieces, last = [], 0
    for m in matches:
        pieces.append(char(s[last:m.start()]))
        last = m.end()
    pieces.append(char(s[last:]))
    results["split"] = Cell.from_list(pieces)
    from ..values import Struct
    group_names = list(rx.groupindex)
    if not group_names:
        names = Struct()
    elif not matches:
        names = Struct(np.empty((0, 0), dtype=object), group_names)
    else:
        sdata = np.empty((1, len(matches)), dtype=object)
        for k, m in enumerate(matches):
            sdata[0, k] = {name: char(m.group(name) or "") for name in group_names}
        names = Struct(sdata, group_names)
    results["names"] = names
    if once:
        if matches:
            results["match"] = char(matches[0].group(0))
            results["tokens"] = tokens_of(matches[0])
            results["start"] = matches[0].start() + 1.0
            results["end"] = float(matches[0].end())
        else:
            results["match"] = char("")
            results["tokens"] = Cell(np.empty((1, 0), dtype=object))
            results["start"] = np.zeros((0, 0))
            results["end"] = np.zeros((0, 0))
    selected = [o for o in opts if o in results]
    order = selected or ["start", "end", "te", "match", "tokens", "names", "split"]
    out = []
    for key in order:
        if key == "te":
            out.append(Cell(np.empty((1, 0), dtype=object)))
        else:
            out.append(results[key])
    n = max(nargout, 1)
    return tuple(out[:n]) if n > 1 else out[0]


