"""Language and text additions.

* Octave-only functions that existing Octave scripts call (``printf``, ``puts``, ``fputs``,
  ``fdisp``, ``ifelse``/``merge``, ``rows``/``columns`` ...).
* The remaining ``mustBe...`` validators and ``MException.last``.
* Text conversion and formatting functions that take char, cellstr and string inputs.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import (
    Cell, MString, StringArray, Struct, char, class_name, is_char, shape_of, to_numeric, to_str,
)
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, is_text, nargs, text

# -- Octave output functions ----------------------------------------------------------------------


def _write(interp, fid: int, s: str) -> None:
    if fid in (1, 2):
        interp.out(s)
    else:
        from .system import file_handle
        file_handle(interp, fid).write(s)


def _fid(v: Any) -> int:
    a = to_numeric(v)
    if a.size != 1:
        raise MatlabError("The file identifier must be a scalar.", "MATLAB:badfid_mx")
    return int(a.flat[0])


@builtin("printf")
def _printf(interp, args, nargout):
    """Octave: printf(template, ...) writes formatted text to the command window."""
    nargs(args, 1, None, "printf")
    return BUILTINS["fprintf"](interp, list(args), nargout)


@builtin("puts")
def _puts(interp, args, nargout):
    """Octave: puts(text) writes text as it is (no newline is added, no format is applied)."""
    nargs(args, 1, 1, "puts")
    interp.out(text(args[0], "text"))
    return None


@builtin("fputs")
def _fputs(interp, args, nargout):
    """Octave: fputs(fid, text) writes text to a file (or to the screen for fid 1 and 2)."""
    nargs(args, 2, 2, "fputs")
    _write(interp, _fid(args[0]), text(args[1], "text"))
    return None


@builtin("fdisp")
def _fdisp(interp, args, nargout):
    """Octave: fdisp(fid, x) writes what disp(x) shows to a file (or to the screen)."""
    nargs(args, 2, 2, "fdisp")
    from ..display import disp_text
    out = disp_text(args[1])
    if not out and is_char(args[1]):
        out = "\n"
    _write(interp, _fid(args[0]), out)
    return None


@builtin("stdout")
def _stdout(interp, args, nargout):
    return 1.0


@builtin("stderr")
def _stderr(interp, args, nargout):
    return 2.0


@builtin("ifelse", "merge")
def _merge(interp, args, nargout):
    """Octave: merge(mask, tval, fval) picks tval where mask is true and fval elsewhere."""
    nargs(args, 3, 3, "merge")
    mask, tval, fval = args
    if not isinstance(mask, np.ndarray) or mask.dtype != bool:
        raise MatlabError("merge: the first argument must be logical.", "Octave:invalid-input-type")
    if mask.size == 1:
        return tval if bool(mask.flat[0]) else fval
    if not (isinstance(tval, np.ndarray) and isinstance(fval, np.ndarray)):
        raise MatlabError("merge: with a mask array, the values must be arrays.", "Octave:invalid-input-type")
    for v in (tval, fval):
        if v.size != 1 and v.shape != mask.shape:
            raise MatlabError("merge: the values must be scalars or have the size of the mask.",
                              "Octave:nonconformant-args")
    t = np.broadcast_to(tval, mask.shape)
    f = np.broadcast_to(fval, mask.shape)
    if t.dtype.kind == "U" or f.dtype.kind == "U":
        out = np.where(mask, t, f).astype("<U1")
    else:
        out = np.where(mask, t, f)
    return out


@builtin("rows")
def _rows(interp, args, nargout):
    nargs(args, 1, 1, "rows")
    return float(shape_of(args[0])[0])


@builtin("columns")
def _columns(interp, args, nargout):
    nargs(args, 1, 1, "columns")
    return float(shape_of(args[0])[1])


@builtin("isargout")
def _isargout(interp, args, nargout):
    nargs(args, 1, 1, "isargout")
    k = to_numeric(args[0])
    n = max(interp.frame.nargout, 1)
    return (k >= 1) & (k <= n)


@builtin("print_usage")
def _print_usage(interp, args, nargout):
    name = interp.frame.name if interp.frame.name != "base" else "function"
    raise MatlabError(f"Invalid call to {name}.", "Octave:invalid-fun-call")


@builtin("usage")
def _usage(interp, args, nargout):
    nargs(args, 1, None, "usage")
    from .strings import matlab_sprintf
    raise MatlabError("usage: " + matlab_sprintf(text(args[0]), list(args[1:])), "Octave:invalid-fun-call")


@builtin("NA")
def _na(interp, args, nargout):
    """Octave's missing-value marker; it behaves like NaN."""
    from ._util import dims_from_args
    dims, _cls = dims_from_args(list(args), "NA")
    return np.full(dims, np.nan)


def _char_test(name: str, pred):
    def f(interp, args, nargout):
        nargs(args, 1, 1, name)
        v = args[0]
        if isinstance(v, Cell):
            data = np.empty(v.data.shape, dtype=object)
            for idx in np.ndindex(v.data.shape):
                data[idx] = f(interp, [v.data[idx]], 1)
            return Cell(data)
        if isinstance(v, MString):
            v = char(str(v))
        if not is_char(v):
            return np.zeros(shape_of(v), dtype=bool)
        out = np.zeros(v.shape, dtype=bool)
        for idx in np.ndindex(v.shape):
            c = v[idx]
            out[idx] = bool(c) and pred(c)
        return out
    builtin(name)(f)


_char_test("isalpha", str.isalpha)
_char_test("isdigit", lambda c: c in "0123456789")
_char_test("isupper", str.isupper)
_char_test("islower", str.islower)
_char_test("ispunct", lambda c: c.isprintable() and not c.isalnum() and not c.isspace())
_char_test("isalnum", str.isalnum)
_char_test("isxdigit", lambda c: c in "0123456789abcdefABCDEF")
_char_test("iscntrl", lambda c: ord(c) < 32 or ord(c) == 127)
_char_test("isgraph", lambda c: c.isprintable() and not c.isspace())
_char_test("isprint", lambda c: c.isprintable())
_char_test("isascii", lambda c: ord(c) < 128)


@builtin("do_string_escapes")
def _do_escapes(interp, args, nargout):
    nargs(args, 1, 1, "do_string_escapes")
    from .strings import matlab_sprintf
    s = text(args[0])
    return char(matlab_sprintf(s.replace("%", "%%"), []))


@builtin("undo_string_escapes")
def _undo_escapes(interp, args, nargout):
    nargs(args, 1, 1, "undo_string_escapes")
    table = {"\n": "\\n", "\t": "\\t", "\r": "\\r", "\a": "\\a", "\b": "\\b", "\f": "\\f", "\v": "\\v",
             "\\": "\\\\", '"': '\\"', "\0": "\\0"}
    return char("".join(table.get(c, c) for c in text(args[0])))


def _pad_impl(pre: bool):
    def f(interp, args, nargout):
        nargs(args, 2, 4, "prepad" if pre else "postpad")
        x = args[0]
        if not isinstance(x, np.ndarray):
            raise MatlabError("The first argument must be an array.")
        n = int(to_numeric(args[1]).flat[0])
        c = to_numeric(args[2]).flat[0] if len(args) > 2 else 0
        if len(args) > 3:
            axis = int(to_numeric(args[3]).flat[0]) - 1
        else:
            axis = next((k for k, d in enumerate(x.shape) if d != 1), 0)
        cur = x.shape[axis]
        if n <= cur:
            index = [slice(None)] * x.ndim
            index[axis] = slice(cur - n, cur) if pre else slice(0, n)
            return x[tuple(index)].copy()
        shape = list(x.shape)
        shape[axis] = n - cur
        if x.dtype.kind == "U":
            fill = np.full(shape, chr(int(c)) if c else "\0", dtype="<U1")
        else:
            fill = np.full(shape, c, dtype=x.dtype if np.asarray(c).dtype.kind != "c" else complex)
        return np.concatenate([fill, x] if pre else [x, fill], axis=axis)
    return f


builtin("postpad")(_pad_impl(False))
builtin("prepad")(_pad_impl(True))


@builtin("sumsq")
def _sumsq(interp, args, nargout):
    nargs(args, 1, 2, "sumsq")
    a = to_numeric(args[0])
    sq = (np.abs(a) ** 2).astype(float)
    return BUILTINS["sum"](interp, [sq, *args[1:]], 1)


@builtin("vec")
def _vec(interp, args, nargout):
    nargs(args, 1, 2, "vec")
    x = args[0]
    if isinstance(x, np.ndarray):
        return x.reshape(-1, 1, order="F")
    if isinstance(x, Cell):
        return Cell(x.data.reshape(-1, 1, order="F"))
    raise MatlabError("vec: the input must be an array.")


def _index_impl(last: bool):
    def f(interp, args, nargout):
        nargs(args, 2, 3, "rindex" if last else "index")
        s, t = text(args[0]), text(args[1])
        direction = text(args[2]).lower() if len(args) > 2 else ("last" if last else "first")
        pos = s.rfind(t) if direction == "last" else s.find(t)
        return float(pos + 1)
    return f


builtin("index")(_index_impl(False))
builtin("rindex")(_index_impl(True))


@builtin("substr")
def _substr(interp, args, nargout):
    """Octave: substr(s, offset, len) is the part of s that starts at offset and has len characters."""
    nargs(args, 2, 3, "substr")
    s = text(args[0])
    offset = int(to_numeric(args[1]).flat[0])
    length = int(to_numeric(args[2]).flat[0]) if len(args) > 2 else len(s) - offset + 1
    if offset < 1 or offset - 1 + length > len(s) or length < 0:
        raise MatlabError("substr: the offset and length are outside the text.", "Octave:index-out-of-bounds")
    return char(s[offset - 1:offset - 1 + length])


@builtin("ostrsplit")
def _ostrsplit(interp, args, nargout):
    """Octave: ostrsplit(s, seps) splits s at every character that is in seps."""
    nargs(args, 2, 3, "ostrsplit")
    s, seps = text(args[0]), text(args[1])
    strip_empty = len(args) > 2 and bool(to_numeric(args[2]).flat[0])
    parts, cur = [], []
    for c in s:
        if c in seps:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    if strip_empty:
        parts = [p for p in parts if p]
    return Cell.from_list([char(p) for p in parts])


@builtin("terminal_size")
def _terminal_size(interp, args, nargout):
    from ..display import FORMAT
    return np.array([[24.0, float(FORMAT.width)]])


@builtin("namelengthmax")
def _namelengthmax(interp, args, nargout):
    return 63.0


# -- MException.last ----------------------------------------------------------------------------------

@builtin("MException.last")
def _mexception_last(interp, args, nargout):
    """MException.last is the error that was caught most recently; MException.last('reset') forgets it."""
    from ..interpreter import make_exception
    from ..objects import MExceptionObject
    if args:
        if text(args[0]).lower() != "reset":
            raise MatlabError("The only option of MException.last is 'reset'.", "MATLAB:MException:InvalidLastArg")
        interp.last_error = None
        return None
    e = interp.last_error
    if e is None:
        return MExceptionObject("", "", Struct(np.empty((0, 1), dtype=object), ["file", "name", "line"]))
    return make_exception(e)


# -- validators that were missing ---------------------------------------------------------------------

def _fail(name: str, message: str):
    raise MatlabError(message, f"MATLAB:validators:{name}")


def _numel(v: Any) -> int:
    n = 1
    for d in shape_of(v):
        n *= d
    return n


@builtin("mustBeFloat")
def _must_be_float(interp, args, nargout):
    nargs(args, 1, 1, "mustBeFloat")
    v = args[0]
    if not (isinstance(v, np.ndarray) and v.dtype.kind in "fc"):
        _fail("mustBeFloat", "Value must be a floating-point array.")
    return None


@builtin("mustBeUnderlyingType")
def _must_be_underlying(interp, args, nargout):
    nargs(args, 2, 2, "mustBeUnderlyingType")
    names = _text_list(args[1])
    if class_name(args[0]) not in names:
        _fail("mustBeUnderlyingType", "Value must have one of these underlying types: " + ", ".join(names) + ".")
    return None


@builtin("mustBeScalarOrEmpty")
def _must_be_scalar_or_empty(interp, args, nargout):
    nargs(args, 1, 1, "mustBeScalarOrEmpty")
    if _numel(args[0]) > 1:
        _fail("mustBeScalarOrEmpty", "Value must be scalar or empty.")
    return None


@builtin("mustBeRow")
def _must_be_row(interp, args, nargout):
    nargs(args, 1, 1, "mustBeRow")
    shape = shape_of(args[0])
    if len(shape) != 2 or shape[0] != 1:
        _fail("mustBeRow", "Value must be a row vector.")
    return None


@builtin("mustBeColumn")
def _must_be_column(interp, args, nargout):
    nargs(args, 1, 1, "mustBeColumn")
    shape = shape_of(args[0])
    if len(shape) != 2 or shape[1] != 1:
        _fail("mustBeColumn", "Value must be a column vector.")
    return None


@builtin("mustBeMatrix")
def _must_be_matrix(interp, args, nargout):
    nargs(args, 1, 1, "mustBeMatrix")
    if len(shape_of(args[0])) != 2:
        _fail("mustBeMatrix", "Value must be a matrix.")
    return None


@builtin("mustBeValidVariableName")
def _must_be_varname(interp, args, nargout):
    nargs(args, 1, 1, "mustBeValidVariableName")
    from ..lexer import KEYWORDS
    v = args[0]
    ok = is_text(v) and (isinstance(v, MString) or v.ndim == 2 and v.shape[0] == 1)
    if ok:
        s = to_str(v)
        ok = bool(s) and (s[0].isalpha() and s.isascii()) and all(c.isalnum() or c == "_" for c in s) \
            and len(s) <= 63 and s not in KEYWORDS
    if not ok:
        _fail("mustBeValidVariableName", "Value must be a valid variable name.")
    return None


@builtin("mustBeNonmissing")
def _must_be_nonmissing(interp, args, nargout):
    nargs(args, 1, 1, "mustBeNonmissing")
    missing = BUILTINS["ismissing"](interp, [args[0]], 1)
    if isinstance(missing, np.ndarray) and missing.any():
        _fail("mustBeNonmissing", "Value must not be missing.")
    return None


@builtin("mustBeNonsparse")
def _must_be_nonsparse(interp, args, nargout):
    nargs(args, 1, 1, "mustBeNonsparse")
    if getattr(args[0], "is_sparse", False) or class_name(args[0]) == "sparse" or hasattr(args[0], "to_dense"):
        _fail("mustBeNonsparse", "Value must not be sparse.")
    return None


@builtin("mustBeSparse")
def _must_be_sparse(interp, args, nargout):
    nargs(args, 1, 1, "mustBeSparse")
    if not hasattr(args[0], "to_dense"):
        _fail("mustBeSparse", "Value must be sparse.")
    return None


@builtin("mustBeBetween")
def _must_be_between(interp, args, nargout):
    nargs(args, 3, 4, "mustBeBetween")
    a, lo, hi = (np.real(to_numeric(x)).astype(float) for x in args[:3])
    kind = text(args[3]).lower() if len(args) > 3 else "closed"
    if kind not in ("closed", "open", "openleft", "closedright", "openright", "closedleft"):
        raise MatlabError(f"Unknown interval type '{kind}'.")
    left_open = kind in ("open", "openleft", "closedright")
    right_open = kind in ("open", "openright", "closedleft")
    ok = np.all((a > lo) if left_open else (a >= lo)) and np.all((a < hi) if right_open else (a <= hi))
    if not ok:
        _fail("mustBeBetween", "Value must be within the allowed interval.")
    return None


def _text_list(v: Any) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x) for x in v.values_f()]
    if isinstance(v, StringArray):
        return [x for x in v.values_f() if x is not None]
    return [text(v)]


# -- is* helpers on text ---------------------------------------------------------------------------------

@builtin("isindex")
def _isindex(interp, args, nargout):
    nargs(args, 1, 2, "isindex")
    v = args[0]
    if not isinstance(v, np.ndarray) or v.dtype.kind == "U":
        return bool_scalar(False)
    if v.dtype == bool:
        ok = True
    else:
        a = np.real(to_numeric(v)).astype(float)
        ok = bool(np.all(np.isfinite(a)) and np.all(a == np.floor(a)) and np.all(a >= 1))
        if ok and len(args) > 1:
            ok = bool(np.all(a <= float(to_numeric(args[1]).flat[0])))
    return bool_scalar(ok)


@builtin("isglobal")
def _isglobal(interp, args, nargout):
    """isglobal(name): true when the variable called name is declared global in this workspace."""
    nargs(args, 1, 1, "isglobal")
    name = text(args[0], "variable name")
    frame = interp.frame
    return bool_scalar(frame.globals.get(name) is interp.global_vars)


# -- regular expression helpers -------------------------------------------------------------------

@builtin("regexptranslate")
def _regexptranslate(interp, args, nargout):
    """regexptranslate(op, text[, replacement]): 'escape', 'wildcard' or 'flexible'."""
    nargs(args, 2, 3, "regexptranslate")
    op = text(args[0]).lower()
    as_string = isinstance(args[1], MString)
    s = text(args[1])
    if op == "escape":
        out = "".join("\\" + c if c in "\\^$.|?*+()[]{}" else c for c in s)
    elif op == "wildcard":
        out = s.replace(".", "\\.").replace("*", ".*").replace("?", ".")
    elif op == "flexible":
        if len(args) < 3:
            raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
        rep = text(args[2])
        out = "".join(rep if not (c.isalnum() or c == "_") else c for c in s)
    else:
        raise MatlabError(f"Unknown regexptranslate operation '{text(args[0])}'. Use 'escape', 'wildcard' or "
                          "'flexible'.", "MATLAB:regexptranslate:InvalidOption")
    return MString(out) if as_string else char(out)


# -- validatestring -------------------------------------------------------------------------------------

@builtin("validatestring")
def _validatestring(interp, args, nargout):
    """validatestring(str, validStrings[, funcName, varName, argIndex]): the valid text that str abbreviates."""
    nargs(args, 2, 5, "validatestring")
    s = args[0]
    if not (isinstance(s, MString) or (is_char(s) and (s.size == 0 or (s.ndim == 2 and s.shape[0] == 1)))):
        raise MatlabError("Expected input to be a character vector or string scalar.",
                          "MATLAB:validatestring:invalidStringInput")
    want = to_str(s)
    valid = args[1]
    if isinstance(valid, StringArray):
        options = [x for x in valid.values_f() if x is not None]
    elif isinstance(valid, MString):
        options = [str(valid)]
    elif isinstance(valid, Cell):
        options = [to_str(x) for x in valid.values_f()]
    elif is_char(valid):
        options = ["".join(r).rstrip() for r in valid] if valid.ndim == 2 and valid.shape[0] > 1 else [to_str(valid)]
    else:
        raise MatlabError("The valid strings must be a cell array of character vectors or a string array.",
                          "MATLAB:validatestring:invalidStringList")
    extra = list(args[2:])
    func = text(extra[0]) if extra and is_text(extra[0]) else ""
    var = text(extra[1]) if len(extra) > 1 and is_text(extra[1]) else ""
    position = next((int(to_numeric(a).flat[0]) for a in extra if not is_text(a)), None)
    subject = "input" if not var else f"input '{var}'"
    if position is not None:
        subject = f"input number {position}, {var}," if var else f"input number {position}"
    listing = ", ".join(f"'{o}'" for o in options)
    head = f"{func}: " if func else ""
    low = want.lower()
    exact = [o for o in options if o.lower() == low]
    matches = [o for o in options if o.lower().startswith(low)] if want else []
    if exact:
        pick = exact[0]
    elif len(matches) == 1:
        pick = matches[0]
    elif matches:
        shortest = min(matches, key=len)
        if not all(m.lower().startswith(shortest.lower()) for m in matches):
            raise MatlabError(f"{head}Expected {subject} to match one of these values:\n\n{listing}\n\n"
                              f"The input, '{want}', matched more than one valid value.",
                              "MATLAB:ambiguousStringChoice")
        pick = shortest
    else:
        raise MatlabError(f"{head}Expected {subject} to match one of these values:\n\n{listing}\n\n"
                          f"The input, '{want}', did not match any of the valid values.",
                          "MATLAB:unrecognizedStringChoice")
    return MString(pick) if isinstance(valid, (MString, StringArray)) else char(pick)


# -- compose ------------------------------------------------------------------------------------------------

_CONVERSION = re.compile(r"%[-+ 0#]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[a-zA-Z]+")


@builtin("compose")
def _compose(interp, args, nargout):
    """compose(format, A1, ..., An): one text per row of the data (or per format), as sprintf writes it."""
    nargs(args, 1, None, "compose")
    from ..values import string_data, string_result
    from .strings import matlab_sprintf
    fmt_v = args[0]
    if isinstance(fmt_v, (MString, StringArray)):
        kind = "string"
    elif isinstance(fmt_v, Cell) or (is_char(fmt_v) and (fmt_v.size == 0 or (fmt_v.ndim == 2 and fmt_v.shape[0] == 1))):
        kind = "cell"
    else:
        raise MatlabError("compose: the format must be text.", "MATLAB:compose:InvalidFormat")
    fdata = string_data(fmt_v)
    formats = list(fdata.flatten(order="F"))
    data = list(args[1:])

    def out_text(items: list, shape: tuple):
        arr = np.empty(len(items), dtype=object)
        for k, x in enumerate(items):
            arr[k] = x if kind == "string" else char(x)
        arr = arr.reshape(shape, order="F")
        return string_result(arr) if kind == "string" else Cell(arr)

    if not data:
        return out_text([matlab_sprintf(f or "", []) for f in formats], fdata.shape)

    def grid_of(a):
        """The elements of a data argument, each as a value that sprintf takes."""
        if isinstance(a, Cell):
            return a.data
        d: np.ndarray
        if isinstance(a, StringArray):
            d = np.empty(a.data.shape, dtype=object)
            for idx in np.ndindex(a.data.shape):
                d[idx] = MString(a.data[idx] if a.data[idx] is not None else "<missing>")
        elif isinstance(a, MString) or (is_char(a) and (a.size == 0 or a.shape[0] == 1)):
            d = np.empty((1, 1), dtype=object)
            d[0, 0] = a
        elif isinstance(a, np.ndarray):
            d = np.empty(a.shape, dtype=object)
            for idx in np.ndindex(a.shape):
                d[idx] = a[idx].reshape(1, 1) if isinstance(a[idx], np.ndarray) else np.array([[a[idx]]])
        else:
            raise MatlabError("compose: unsupported data argument.", "MATLAB:compose:InvalidInput")
        if d.ndim > 2:
            raise MatlabError("compose: the data arguments must be matrices.", "MATLAB:compose:InvalidInput")
        return d

    grids = [grid_of(a) for a in data]
    nrows = max([g.shape[0] for g in grids] + [fdata.shape[0]])
    for g in grids:
        if g.shape[0] not in (1, nrows):
            raise MatlabError("compose: the data arguments must have the same number of rows.",
                              "MATLAB:compose:SizeMismatch")

    def row_values(i: int) -> list:
        return [g[i if g.shape[0] > 1 else 0, c] for g in grids for c in range(g.shape[1])]

    if len(formats) > 1:
        # one text per format (and per row of data)
        out = np.empty((nrows, fdata.shape[1]), dtype=object)
        for idx in np.ndindex(out.shape):
            f = fdata[idx[0] if fdata.shape[0] > 1 else 0, idx[1]]
            out[idx] = matlab_sprintf(f or "", row_values(idx[0]))
        return out_text(list(out.flatten(order="F")), out.shape)
    # one format: each row of the data gives one row of texts; when a row has more values than
    # the format has conversions, the format is used again for the next column of the result
    fmt = formats[0] or ""
    nops = max(len(_CONVERSION.findall(fmt.replace("%%", ""))), 1)
    rows = [row_values(i) for i in range(nrows)]
    ncols = max(1, -(-len(rows[0]) // nops)) if rows else 1
    out = np.empty((nrows, ncols), dtype=object)
    for i, vals in enumerate(rows):
        for j in range(ncols):
            out[i, j] = matlab_sprintf(fmt, vals[j * nops:(j + 1) * nops])
    return out_text(list(out.flatten(order="F")), out.shape)


# -- rearranging string arrays ---------------------------------------------------------------------------------

def _string_shim(name: str) -> None:
    """Let a function that rearranges array elements (repmat, fliplr ...) work on string arrays."""
    original = BUILTINS.get(name)
    if original is None:
        return

    def f(interp, args, nargout, _fn=original):
        if not args or not isinstance(args[0], (MString, StringArray)):
            return _fn(interp, args, nargout)
        from ..values import string_data, string_result
        data = string_data(args[0])
        index = np.arange(1.0, data.size + 1).reshape(data.shape, order="F")
        moved = _fn(interp, [index, *args[1:]], 1)
        if isinstance(moved, tuple):
            moved = moved[0]
        flat = data.flatten(order="F")
        out = np.empty(moved.shape, dtype=object)
        for idx in np.ndindex(moved.shape):
            out[idx] = flat[int(moved[idx]) - 1]
        return string_result(out) if out.size != 0 else StringArray(out)

    f.__module__ = __name__
    BUILTINS[name] = f


for _name in ("repmat", "fliplr", "flipud", "flip", "rot90", "circshift", "permute", "squeeze", "repelem",
              "reshape", "transpose", "ctranspose"):
    _string_shim(_name)


_classic_num2cell = BUILTINS.get("num2cell")


@builtin("num2cell")
def _num2cell(interp, args, nargout):
    if args and isinstance(args[0], (MString, StringArray)) and len(args) == 1:
        from ..values import string_data
        data = string_data(args[0])
        out = np.empty(data.shape, dtype=object)
        for idx in np.ndindex(data.shape):
            x = data[idx]
            if x is None:
                d = np.empty((1, 1), dtype=object)
                d[0, 0] = None
                out[idx] = StringArray(d)
            else:
                out[idx] = MString(x)
        return Cell(out)
    return _classic_num2cell(interp, args, nargout)


def _string_extreme(name: str, pick_last: bool) -> None:
    """max / min of string arrays: the text that sorts last / first."""
    original = BUILTINS.get(name)
    if original is None:
        return

    def f(interp, args, nargout, _fn=original):
        if not args or not isinstance(args[0], (MString, StringArray)):
            return _fn(interp, args, nargout)
        from ..values import string_data, string_result
        choose = max if pick_last else min
        if len(args) > 1 and isinstance(args[1], (MString, StringArray)):
            a, b = np.broadcast_arrays(string_data(args[0]), string_data(args[1]))
            out = np.empty(a.shape, dtype=object)
            for idx in np.ndindex(a.shape):
                x, y = a[idx], b[idx]
                out[idx] = None if x is None or y is None else choose(x, y)
            return string_result(out)
        data = string_data(args[0])
        dim = None
        if len(args) > 2 and not is_text(args[2]):
            dim = int(to_numeric(args[2]).flat[0]) - 1
        if dim is None:
            dim = next((k for k, d in enumerate(data.shape) if d != 1), 0)
        moved = np.moveaxis(data, dim, 0)
        res = np.empty(moved.shape[1:], dtype=object)
        where = np.zeros(moved.shape[1:])
        for idx in np.ndindex(moved.shape[1:]):
            col = [(x, k) for k, x in enumerate(moved[(slice(None),) + idx]) if x is not None]
            if not col:
                res[idx], where[idx] = None, 1
                continue
            best = choose(col, key=lambda p: p[0])
            first = next(p for p in col if p[0] == best[0])
            res[idx], where[idx] = first[0], first[1] + 1
        out = string_result(np.expand_dims(res, dim))
        return (out, np.expand_dims(where, dim)) if nargout > 1 else out

    f.__module__ = __name__
    BUILTINS[name] = f


_string_extreme("max", True)
_string_extreme("min", False)
