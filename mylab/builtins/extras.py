"""More core functions: indexing helpers, text utilities, validation, special functions, geometry."""

from __future__ import annotations

import itertools
import re

import numpy as np

from .._lazy import lazy_function, lazy_module
from ..errors import MatlabError
from ..values import (
    INT_TYPES, Cell, MString, Struct, char, class_name, codes_to_char, is_char,
    saturate_int, simplify, to_int, to_str,
)
from ._registry import builtin
from ._util import bool_scalar, fnum, get_dim, is_text, nargs, num, text

sp = lazy_module("scipy.special", globals(), "sp")  # imported on first use: keeps start-up fast


def _given(args: list, i: int) -> bool:
    return len(args) > i and not (isinstance(args[i], np.ndarray) and args[i].size == 0 and not is_text(args[i]))


def _scalar(v, what="argument") -> float:
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"Expected {what} to be a scalar.")
    return float(np.real(a.flat[0]))


def _bcast(name: str, *vals) -> list[np.ndarray]:
    arrs = [fnum(v) for v in vals]
    shapes = {a.shape for a in arrs if a.size != 1}
    if len(shapes) > 1:
        raise MatlabError(f"{name}: inputs must be the same size or scalars.")
    return [np.array(a) for a in np.broadcast_arrays(*arrs)]


# -- index conversion ------------------------------------------------------------------

def _size_vec(v) -> list[int]:
    s = fnum(v).flatten()
    if s.size < 1 or np.any(s < 0) or np.any(s != np.round(s)):
        raise MatlabError("Size vector must contain non-negative integers.")
    return [int(x) for x in s]


def _eff_dims(sz: list[int], k: int) -> list[int]:
    """Dimensions as seen by k subscripts: the last one absorbs the trailing dims."""
    if k >= len(sz):
        return sz + [1] * (k - len(sz))
    return sz[:k - 1] + [int(np.prod(sz[k - 1:]))]


@builtin("sub2ind")
def _sub2ind(interp, args, nargout):
    nargs(args, 2, None, "sub2ind")
    sz = _size_vec(args[0])
    subs = _bcast("sub2ind", *args[1:])
    dims = _eff_dims(sz, len(subs))
    out = np.zeros(subs[0].shape)
    mult = 1
    for s, d in zip(subs, dims):
        if np.any(s != np.round(s)) or np.any(s < 1):
            raise MatlabError("Subscript indices must be positive integers.")
        if np.any(s > d):
            raise MatlabError("Out of range subscript.")
        out = out + (s - 1) * mult
        mult *= d
    return out + 1


@builtin("ind2sub")
def _ind2sub(interp, args, nargout):
    nargs(args, 2, 2, "ind2sub")
    sz = _size_vec(args[0])
    ind = fnum(args[1])
    if np.any(ind != np.round(ind)) or np.any(ind < 1):
        raise MatlabError("Index must be a positive integer.")
    if ind.size and ind.max() > np.prod(sz):
        raise MatlabError("Index out of range.")
    nout = max(nargout, 1)
    dims = _eff_dims(sz, nout)
    rem = ind - 1
    outs = []
    for d in dims[:-1]:
        outs.append(np.mod(rem, d) + 1)
        rem = np.floor(rem / d)
    outs.append(rem + 1)
    return tuple(outs) if nout > 1 else outs[0]


@builtin("perms")
def _perms(interp, args, nargout):
    nargs(args, 1, 1, "perms")
    v = args[0]
    if isinstance(v, Cell):
        items = v.data.flatten(order="F")
        n = items.size
        rows = list(itertools.permutations(range(n)))[::-1]
        data = np.empty((len(rows), n), dtype=object)
        for i, r in enumerate(rows):
            for j, k in enumerate(r):
                data[i, j] = items[k]
        return Cell(data)
    if not isinstance(v, np.ndarray):
        raise MatlabError("perms: input must be an array.")
    flat = v.flatten(order="F")
    n = flat.size
    if n == 0:
        return np.zeros((1, 0), dtype=v.dtype)
    if n > 11:
        raise MatlabError("perms: too many elements (the result would be enormous).")
    idx = np.array(list(itertools.permutations(range(n)))[::-1], dtype=int)
    return flat[idx]


@builtin("bsxfun")
def _bsxfun(interp, args, nargout):
    nargs(args, 3, 3, "bsxfun")
    f, a, b = args
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise MatlabError("bsxfun: inputs must be arrays.")
    for da, db in zip(a.shape, b.shape):
        if da != db and da != 1 and db != 1:
            raise MatlabError("bsxfun: non-singleton dimensions of the two inputs must match.")
    shape = tuple(max(da, db) if 0 not in (da, db) else 0 for da, db in zip(a.shape, b.shape))
    ax = np.broadcast_to(a, shape).copy()
    bx = np.broadcast_to(b, shape).copy()
    vals = interp.call_function_value(f, [ax, bx], 1)
    if not vals:
        raise MatlabError("bsxfun: the function must return a value.")
    return vals[0]


@builtin("idivide")
def _idivide(interp, args, nargout):
    nargs(args, 2, 3, "idivide")
    a, b = args[0], args[1]
    opt = text(args[2]).lower() if len(args) > 2 else "fix"
    ints = [x for x in (a, b) if isinstance(x, np.ndarray) and x.dtype.kind in "iu"]
    if not ints:
        raise MatlabError("idivide: at least one input must be an integer type.")
    if len(ints) == 2 and a.dtype != b.dtype:
        raise MatlabError("idivide: integer inputs must be of the same class.")
    cls = ints[0].dtype
    av, bv = _bcast("idivide", a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = av.astype(float) / bv.astype(float)
    rounder = {"fix": np.trunc, "floor": np.floor, "ceil": np.ceil,
               "round": lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5)}.get(opt)
    if rounder is None:
        raise MatlabError(f"idivide: unknown rounding option '{opt}'.")
    with np.errstate(invalid="ignore"):
        r = rounder(q)
    return saturate_int(r, cls.type)


@builtin("vecnorm")
def _vecnorm(interp, args, nargout):
    nargs(args, 1, 3, "vecnorm")
    a = fnum(args[0])
    p = _scalar(args[1]) if _given(args, 1) else 2.0
    if not p > 0:
        raise MatlabError("vecnorm: p must be a positive real scalar or Inf.")
    single = a.dtype in (np.float32, np.complex64)
    axis = get_dim(args, 2, a)
    m = np.abs(a.astype(complex if a.dtype.kind == "c" else float))
    if axis >= m.ndim:
        r = m  # along a dimension of length one every norm is the absolute value
    elif m.shape[axis] == 0:
        shape = list(m.shape)
        shape[axis] = 1
        r = np.zeros(shape)
    elif np.isinf(p):
        r = m.max(axis=axis, keepdims=True)
    elif p == 2:
        scale = m.max(axis=axis, keepdims=True)
        safe = np.where((scale > 0) & np.isfinite(scale), scale, 1.0)
        r = safe * np.sqrt(((m / safe) ** 2).sum(axis=axis, keepdims=True))
        r = np.where(np.isfinite(scale), r, scale)
    elif p == 1:
        r = m.sum(axis=axis, keepdims=True)
    else:
        r = (m ** p).sum(axis=axis, keepdims=True) ** (1 / p)
    from ..values import as_2d
    r = as_2d(np.asarray(r))
    return r.astype(np.float32) if single else r


@builtin("nthargout")
def _nthargout(interp, args, nargout):
    nargs(args, 2, None, "nthargout")
    n = fnum(args[0]).flatten().astype(int)
    rest = list(args[1:])
    if isinstance(rest[0], np.ndarray) and not is_char(rest[0]):
        total = to_int(rest.pop(0), "number of outputs")
    else:
        total = int(n.max())
    f = rest[0]
    if np.any(n < 1) or np.any(n > total):
        raise MatlabError("nthargout: output index out of range.")
    vals = interp.call_function_value(f, rest[1:], total)
    if len(vals) < total:
        raise MatlabError("nthargout: the function returned fewer outputs than requested.")
    if n.size == 1:
        return vals[n[0] - 1]
    return Cell.from_list([vals[k - 1] for k in n])


# -- sscanf ---------------------------------------------------------------------------------

_NUM_RE = {
    "d": r"[+-]?\d+",
    "i": r"[+-]?(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9]\d*)",
    "u": r"[+-]?\d+",
    "o": r"[0-7]+",
    "x": r"(?:0[xX])?[0-9a-fA-F]+",
    "f": r"[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?|[nN][aA][nN])",
}
for _c in "eEgG":
    _NUM_RE[_c] = _NUM_RE["f"]
_NUM_RE["X"] = _NUM_RE["x"]


def _parse_scan_format(fmt: str) -> list[tuple]:
    toks: list[tuple] = []
    i = 0
    while i < len(fmt):
        c = fmt[i]
        if c.isspace():
            while i < len(fmt) and fmt[i].isspace():
                i += 1
            toks.append(("ws",))
            continue
        if c == "%":
            m = re.match(r"%(\*?)(\d*)(?:l|h)?([diuoxXfeEgGsc%]|\[[^\]]*\])", fmt[i:])
            if not m:
                raise MatlabError(f"sscanf: invalid format near '{fmt[i:]}'.")
            skip, width, kind = m.group(1) == "*", m.group(2), m.group(3)
            i += m.end()
            if kind == "%":
                toks.append(("lit", "%"))
            else:
                toks.append(("conv", kind, int(width) if width else None, skip))
            continue
        j = i
        while j < len(fmt) and not fmt[j].isspace() and fmt[j] != "%":
            j += 1
        toks.append(("lit", fmt[i:j]))
        i = j
    return toks


def _scan(s: str, fmt: str):
    toks = _parse_scan_format(fmt)
    values: list[float] = []
    kinds: list[str] = []
    count = 0
    pos = 0
    errmsg = ""
    if not any(t[0] == "conv" for t in toks):
        return values, kinds, count, errmsg
    while pos < len(s):
        start = pos
        stop = False
        for t in toks:
            if t[0] == "ws":
                while pos < len(s) and s[pos].isspace():
                    pos += 1
                continue
            if t[0] == "lit":
                if s.startswith(t[1], pos):
                    pos += len(t[1])
                    continue
                stop = True
                break
            _, kind, width, skip = t
            if kind not in ("c",) and not kind.startswith("["):
                while pos < len(s) and s[pos].isspace():
                    pos += 1
            if pos >= len(s):
                stop = True
                break
            seg = s[pos:pos + width] if width else s[pos:]
            if kind == "s":
                m = re.match(r"\S+", seg)
                if not m:
                    stop = True
                    break
                word = m.group(0)
                pos += len(word)
                if not skip:
                    values.extend(float(ord(ch)) for ch in word)
                    kinds.extend("s" * len(word))
                    count += 1
            elif kind == "c":
                w = width or 1
                chunk = s[pos:pos + w]
                pos += len(chunk)
                if not skip:
                    values.extend(float(ord(ch)) for ch in chunk)
                    kinds.extend("c" * len(chunk))
                    count += 1
            elif kind.startswith("["):
                body = kind[1:-1]
                cls = "[" + body + "]+"
                m = re.match(cls, seg)
                if not m:
                    stop = True
                    break
                pos += m.end()
                if not skip:
                    values.extend(float(ord(ch)) for ch in m.group(0))
                    kinds.extend("s" * m.end())
                    count += 1
            else:
                m = re.match(_NUM_RE[kind], seg)
                if not m or m.group(0) in ("+", "-"):
                    stop = True
                    errmsg = "Matching failure in format."
                    break
                tok = m.group(0)
                pos += len(tok)
                if kind in "fFeEgG":
                    v = float(tok)
                elif kind == "o":
                    v = float(int(tok, 8))
                elif kind in "xX":
                    v = float(int(tok, 16))
                elif kind == "i":
                    v = float(int(tok, 0)) if not re.fullmatch(r"[+-]?0[0-7]+", tok) else float(int(tok, 8))
                else:
                    v = float(int(tok))
                if not skip:
                    values.append(v)
                    kinds.append("n")
                    count += 1
        if stop or pos == start:
            break
    return values, kinds, count, errmsg


@builtin("sscanf")
def _sscanf(interp, args, nargout):
    nargs(args, 2, 3, "sscanf")
    s = text(args[0], "input text")
    fmt = text(args[1], "format")
    from .strings import process_escapes
    fmt = process_escapes(fmt)
    values, kinds, count, errmsg = _scan(s, fmt)
    all_text = bool(kinds) and all(k in "sc" for k in kinds)
    arr = np.array(values, dtype=float)
    if len(args) > 2:
        sz = fnum(args[2]).flatten()
        if sz.size == 1:
            n = arr.size if np.isinf(sz[0]) else int(sz[0])
            out = arr[:n].reshape(-1, 1)
        else:
            m = int(sz[0])
            ncol = int(np.ceil(arr.size / m)) if np.isinf(sz[1]) else int(sz[1])
            buf = np.zeros(m * ncol)
            take = min(arr.size, m * ncol)
            buf[:take] = arr[:take]
            out = buf.reshape((m, ncol), order="F")
    else:
        out = arr.reshape(-1, 1)
        if all_text:
            out = arr.reshape(1, -1)
    if all_text:
        out = codes_to_char(out)
    elif arr.size == 0:
        out = np.zeros((0, 0))
    outs = (out, float(count), char(errmsg), float(len(s) + 1))
    return outs[:max(nargout, 1)] if nargout > 1 else out


# -- validation -----------------------------------------------------------------------------

_NUMERIC_CLASSES = {"double", "single", *INT_TYPES}


def _class_ok(v, cls: str) -> bool:
    c = class_name(v)
    if cls == c:
        return True
    if hasattr(v, "isa_class"):  # objects: the class itself, a superclass, or a category
        return bool(v.isa_class(cls))
    if cls == "numeric":
        return c in _NUMERIC_CLASSES
    if cls == "float":
        return c in ("double", "single")
    if cls == "integer":
        return c in INT_TYPES
    return False


def _cell_texts(v) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x) for x in v.values_f()]
    if is_text(v):
        return [text(v)]
    raise MatlabError("Expected a cell array of character vectors.")


@builtin("validateattributes")
def _validateattributes(interp, args, nargout):
    nargs(args, 3, 6, "validateattributes")
    v, classes, attrs = args[0], args[1], args[2]
    func = text(args[3]) if len(args) > 3 and is_text(args[3]) else ""
    var = text(args[4]) if len(args) > 4 and is_text(args[4]) else "input"
    if len(args) > 3 and not is_text(args[3]) and _given(args, 3):
        var = f"input number {to_int(args[3])}"
    if len(args) > 5:
        var = f"input number {to_int(args[5])}, {var},"
    prefix = f"{func}: " if func else ""

    def fail(msg):
        raise MatlabError(f"{prefix}Expected {var} to be {msg}.")

    cls_list = _cell_texts(classes) if not (isinstance(classes, Cell) and classes.data.size == 0) else []
    if cls_list and not any(_class_ok(v, c) for c in cls_list):
        fail("one of these types: " + ", ".join(cls_list) + f" (instead its type was {class_name(v)})")
    if not isinstance(attrs, Cell):
        raise MatlabError("validateattributes: attributes must be a cell array.")
    items = attrs.values_f()
    shape = v.shape if isinstance(v, (np.ndarray, Cell, Struct)) else (1, 1)
    size = int(np.prod(shape))
    isnum = isinstance(v, np.ndarray)
    a = fnum(v) if isnum else None
    k = 0

    def numeric_needed(name):
        if a is None:
            fail(f"{name} (it is not numeric)")
        return a

    while k < len(items):
        name = to_str(items[k]).lower()
        k += 1
        if name in (">", ">=", "<", "<=", "numel", "size", "ncols", "nrows", "ndims"):
            if k >= len(items):
                raise MatlabError(f"validateattributes: attribute '{name}' needs a value.")
            arg = items[k]
            k += 1
        if name == "scalar":
            if size != 1:
                fail("scalar")
        elif name == "vector":
            if not (len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)):
                fail("a vector")
        elif name == "row":
            if not (len(shape) == 2 and shape[0] == 1):
                fail("a row vector")
        elif name == "column":
            if not (len(shape) == 2 and shape[1] == 1):
                fail("a column vector")
        elif name in ("2d", "matrix"):
            if len(shape) != 2:
                fail("two-dimensional")
        elif name == "square":
            if not (len(shape) == 2 and shape[0] == shape[1]):
                fail("square")
        elif name == "nonempty":
            if size == 0:
                fail("nonempty")
        elif name in ("nonsparse", "3d", "scalartext"):
            pass
        elif name == "real":
            if numeric_needed("real") is not None and np.iscomplexobj(a) and np.any(a.imag):
                fail("real")
        elif name == "finite":
            if not np.all(np.isfinite(numeric_needed("finite"))):
                fail("finite")
        elif name == "nonnan":
            if np.any(np.isnan(numeric_needed("non-NaN"))):
                fail("non-NaN")
        elif name == "integer":
            x = numeric_needed("integer-valued")
            if not np.all(np.isfinite(x) & (x == np.round(x))):
                fail("integer-valued")
        elif name == "positive":
            if not np.all(np.real(numeric_needed("positive")) > 0):
                fail("positive")
        elif name == "nonnegative":
            if not np.all(np.real(numeric_needed("nonnegative")) >= 0):
                fail("nonnegative")
        elif name == "nonzero":
            if not np.all(numeric_needed("nonzero") != 0):
                fail("nonzero")
        elif name == "binary":
            x = numeric_needed("binary")
            if not np.all((x == 0) | (x == 1)):
                fail("binary")
        elif name == "even":
            x = numeric_needed("even")
            if not np.all(np.mod(x, 2) == 0):
                fail("even")
        elif name == "odd":
            x = numeric_needed("odd")
            if not np.all(np.mod(x, 2) == 1):
                fail("odd")
        elif name in ("increasing", "decreasing", "nondecreasing", "nonincreasing"):
            d = np.diff(np.real(numeric_needed(name)).flatten())
            ok = {"increasing": np.all(d > 0), "decreasing": np.all(d < 0),
                  "nondecreasing": np.all(d >= 0), "nonincreasing": np.all(d <= 0)}[name]
            if not ok or (a.size and np.any(np.isnan(a))):
                fail(name)
        elif name in (">", ">=", "<", "<="):
            x = np.real(numeric_needed("a number"))
            lim = _scalar(arg)
            ops = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}
            word = {">": "greater than", ">=": "greater than or equal to", "<": "less than",
                    "<=": "less than or equal to"}[name]
            if not np.all(ops[name](x, lim)):
                fail(f"an array with all of the values {word} {lim:g}")
        elif name == "numel":
            if size != to_int(arg):
                fail(f"an array with {to_int(arg)} elements")
        elif name == "ncols":
            if shape[1] != to_int(arg):
                fail(f"an array with {to_int(arg)} columns")
        elif name == "nrows":
            if shape[0] != to_int(arg):
                fail(f"an array with {to_int(arg)} rows")
        elif name == "ndims":
            if len(shape) != to_int(arg):
                fail(f"an array with {to_int(arg)} dimensions")
        elif name == "size":
            want = fnum(arg).flatten()
            got = list(shape) + [1] * max(0, want.size - len(shape))
            if len(got) != want.size or any(not np.isnan(w) and int(w) != g for w, g in zip(want, got)):
                fail("of size " + "x".join("N" if np.isnan(w) else str(int(w)) for w in want))
        else:
            raise MatlabError(f"validateattributes: unknown attribute '{name}'.")
    return None


# -- text utilities ----------------------------------------------------------------------

def _tmap(v, fn):
    """Apply str->value to char row, string scalar, or each element of a cell array."""
    if isinstance(v, Cell):
        data = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            data[idx] = _tmap(v.data[idx], fn)
        return Cell(data)
    if isinstance(v, MString):
        r = fn(str(v))
        return MString(r) if isinstance(r, str) else r
    if is_char(v):
        if v.shape[0] > 1:
            rows = [fn("".join(r)) for r in v]
            if not all(isinstance(r, str) for r in rows):
                raise MatlabError("Expected a character row vector.")
            w = max(len(r) for r in rows)
            return np.array([list(r.ljust(w)) for r in rows], dtype="<U1").reshape(len(rows), w)
        r = fn(to_str(v))
        return char(r) if isinstance(r, str) else r
    raise MatlabError("Expected text input (character vector, string or cell array of text).")


def _tcount(v, fn) -> np.ndarray:
    if isinstance(v, Cell):
        out = np.zeros(v.shape)
        for idx in np.ndindex(v.shape):
            out[idx] = fn(to_str(v.data[idx]))
        return out
    return np.array([[float(fn(text(v)))]])


def _patterns(p) -> list[str]:
    if isinstance(p, Cell):
        return [to_str(x) for x in p.values_f()]
    return [text(p, "pattern")]


def _ignore_case(args, start) -> bool:
    k = start
    while k < len(args) - 1:
        if is_text(args[k]) and text(args[k]).lower() == "ignorecase":
            return bool(fnum(args[k + 1]).flat[0])
        k += 2
    return False


def _find_first(s: str, pats: list[str]) -> tuple[int, int] | None:
    best = None
    for p in pats:
        i = s.find(p)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, len(p))
    return best


@builtin("strjust")
def _strjust(interp, args, nargout):
    nargs(args, 1, 2, "strjust")
    mode = text(args[1]).lower() if len(args) > 1 else "right"
    v = args[0]
    if isinstance(v, Cell) or isinstance(v, MString):
        raise MatlabError("strjust: input must be a character array.")
    if not is_char(v) or v.size == 0:
        return v
    w = v.shape[1]
    rows = []
    for r in v:
        s = "".join(r).strip(" ")
        if mode == "right":
            rows.append(s.rjust(w))
        elif mode == "left":
            rows.append(s.ljust(w))
        elif mode == "center":
            total = w - len(s)
            left = total // 2
            rows.append(" " * left + s + " " * (total - left))
        else:
            raise MatlabError("strjust: mode must be 'left', 'right' or 'center'.")
    return np.array([list(r) for r in rows], dtype="<U1").reshape(v.shape)


@builtin("pad")
def _pad(interp, args, nargout):
    nargs(args, 1, 4, "pad")
    v = args[0]
    rest = list(args[1:])
    n = None
    if rest and not is_text(rest[0]):
        n = to_int(rest.pop(0), "length")
    side = "right"
    fill = " "
    if rest:
        side = text(rest.pop(0)).lower()
    if rest:
        fill = text(rest.pop(0))
        if len(fill) != 1:
            raise MatlabError("pad: pad character must be a single character.")
    if side not in ("left", "right", "both"):
        raise MatlabError("pad: side must be 'left', 'right' or 'both'.")
    if n is None:
        if isinstance(v, Cell):
            n = max((len(to_str(x)) for x in v.values_f()), default=0)
        else:
            n = len(text(v))

    def fn(s):
        extra = max(0, n - len(s))
        if side == "left":
            return fill * extra + s
        if side == "right":
            return s + fill * extra
        left = extra // 2
        return fill * left + s + fill * (extra - left)
    return _tmap(v, fn)


def _pos(v) -> int | None:
    if isinstance(v, np.ndarray) and not is_char(v):
        return to_int(v, "position")
    return None


@builtin("extractBefore")
def _extract_before(interp, args, nargout):
    nargs(args, 2, 2, "extractBefore")
    p = _pos(args[1])
    if p is not None:
        def fn(s):
            if p < 1 or p > len(s) + 1:
                raise MatlabError("extractBefore: position is out of range.")
            return s[:p - 1]
    else:
        pats = _patterns(args[1])

        def fn(s):
            hit = _find_first(s, pats)
            return s[:hit[0]] if hit else ""
    return _tmap(args[0], fn)


@builtin("extractAfter")
def _extract_after(interp, args, nargout):
    nargs(args, 2, 2, "extractAfter")
    p = _pos(args[1])
    if p is not None:
        def fn(s):
            if p < 0 or p > len(s):
                raise MatlabError("extractAfter: position is out of range.")
            return s[p:]
    else:
        pats = _patterns(args[1])

        def fn(s):
            hit = _find_first(s, pats)
            return s[hit[0] + hit[1]:] if hit else ""
    return _tmap(args[0], fn)


def _insert(after: bool, name: str):
    def f(interp, args, nargout):
        nargs(args, 3, 3, name)
        new = text(args[2], "new text")
        p = _pos(args[1])
        if p is not None:
            def fn(s):
                cut = p if after else p - 1
                if cut < 0 or cut > len(s):
                    raise MatlabError(f"{name}: position is out of range.")
                return s[:cut] + new + s[cut:]
            return _tmap(args[0], fn)
        pats = [x for x in _patterns(args[1]) if x]
        if not pats:
            return args[0]
        rx = re.compile("|".join(re.escape(x) for x in sorted(pats, key=len, reverse=True)))
        return _tmap(args[0], lambda s: rx.sub(lambda m: m.group(0) + new if after else new + m.group(0), s))
    return f


builtin("insertAfter")(_insert(True, "insertAfter"))
builtin("insertBefore")(_insert(False, "insertBefore"))


def _split_one(s: str, delims: list[str] | None, collapse: bool) -> list[str]:
    if delims is None:
        parts = re.split(r"\s+" if collapse else r"\s", s)
        return parts
    rx = "|".join(re.escape(d) for d in sorted(delims, key=len, reverse=True))
    if collapse:
        rx = f"(?:{rx})+"
    return re.split(rx, s)


_SPEC_RE = re.compile(r"%(?!%)[-+ #0]*(\*|\d+)?(?:\.(\*|\d+))?[diouxXfeEgGcs]")


# -- cells ------------------------------------------------------------------------------------

@builtin("mat2cell")
def _mat2cell(interp, args, nargout):
    nargs(args, 2, 3, "mat2cell")
    a = args[0]
    shape = a.shape if isinstance(a, (np.ndarray, Cell)) else (1, 1)
    rdiv = fnum(args[1]).flatten().astype(int)
    cdiv = fnum(args[2]).flatten().astype(int) if len(args) > 2 else np.array([shape[1]])
    if rdiv.sum() != shape[0] or cdiv.sum() != shape[1]:
        raise MatlabError("mat2cell: dimension vectors must sum to the size of the input.")
    out = np.empty((rdiv.size, cdiv.size), dtype=object)
    r0 = 0
    for i, rn in enumerate(rdiv):
        c0 = 0
        for j, cn in enumerate(cdiv):
            if isinstance(a, Cell):
                out[i, j] = Cell(a.data[r0:r0 + rn, c0:c0 + cn].copy())
            else:
                out[i, j] = a[r0:r0 + rn, c0:c0 + cn].copy()
            c0 += cn
        r0 += rn
    return Cell(out)


@builtin("celldisp")
def _celldisp(interp, args, nargout):
    nargs(args, 1, 2, "celldisp")
    from ..display import format_display
    c = args[0]
    if not isinstance(c, Cell):
        raise MatlabError("celldisp: input must be a cell array.")
    name = text(args[1]) if len(args) > 1 else "C"

    def show(cell: Cell, prefix: str):
        vec = cell.shape[0] == 1 or cell.shape[1] == 1
        for k, idx in enumerate(itertools.product(range(cell.shape[1]), range(cell.shape[0]))):
            j, i = idx
            v = cell.data[i, j]
            label = f"{prefix}{{{k + 1}}}" if vec else f"{prefix}{{{i + 1},{j + 1}}}"
            if isinstance(v, Cell) and v.data.size:
                show(v, label)
            else:
                interp.out(format_display(label, v))
    show(c, name)
    return None


# -- matrix structure tests ----------------------------------------------------------------

def _square_num(v) -> np.ndarray | None:
    if not isinstance(v, np.ndarray) or is_char(v):
        return None
    return num(v)


@builtin("isdiag")
def _isdiag(interp, args, nargout):
    nargs(args, 1, 1, "isdiag")
    a = _square_num(args[0])
    if a is None:
        return bool_scalar(False)
    return bool_scalar(bool(np.all(a[~np.eye(*a.shape, dtype=bool)] == 0)))


@builtin("istriu")
def _istriu(interp, args, nargout):
    nargs(args, 1, 1, "istriu")
    a = _square_num(args[0])
    return bool_scalar(a is not None and bool(np.all(np.tril(a, -1) == 0)))


@builtin("istril")
def _istril(interp, args, nargout):
    nargs(args, 1, 1, "istril")
    a = _square_num(args[0])
    return bool_scalar(a is not None and bool(np.all(np.triu(a, 1) == 0)))


def _sym_check(conj: bool, name: str):
    def f(interp, args, nargout):
        nargs(args, 1, 2, name)
        a = _square_num(args[0])
        if a is None or a.ndim != 2 or a.shape[0] != a.shape[1]:
            return bool_scalar(False)
        skew = False
        tol = 0.0
        if len(args) > 1:
            if is_text(args[1]):
                opt = text(args[1]).lower()
                if opt not in ("skew", "nonskew"):
                    raise MatlabError(f"{name}: option must be 'skew' or 'nonskew'.")
                skew = opt == "skew"
            else:
                tol = _scalar(args[1], "tolerance")
        t = a.conj().T if conj else a.T
        target = -t if skew else t
        if tol == 0:
            return bool_scalar(bool(np.all(a == target)))
        scale = np.linalg.norm(a, np.inf) if a.size else 0.0
        return bool_scalar(bool(np.linalg.norm(a - target, np.inf) <= tol * scale))
    return f


builtin("issymmetric")(_sym_check(False, "issymmetric"))
builtin("ishermitian")(_sym_check(True, "ishermitian"))


# -- angles and coordinates ------------------------------------------------------------------

def _wrap_positive(x: np.ndarray, period: float) -> np.ndarray:
    y = np.mod(x, period)
    return np.where((y == 0) & (x > 0), period, y)


def _wrap_centered(x: np.ndarray, half: float) -> np.ndarray:
    out = x.copy()
    q = (x < -half) | (x > half)
    out[q] = _wrap_positive(x[q] + half, 2 * half) - half
    return out


builtin("wrapTo360")(lambda interp, args, nargout: _wrap_positive(fnum(args[0]).astype(float), 360.0))
builtin("wrapTo2Pi")(lambda interp, args, nargout: _wrap_positive(fnum(args[0]).astype(float), 2 * np.pi))
builtin("wrapTo180")(lambda interp, args, nargout: _wrap_centered(fnum(args[0]).astype(float), 180.0))
builtin("wrapToPi")(lambda interp, args, nargout: _wrap_centered(fnum(args[0]).astype(float), np.pi))


def _split_matrix(args, n, name):
    if len(args) == 1:
        p = fnum(args[0])
        if p.ndim != 2 or p.shape[1] not in n:
            raise MatlabError(f"{name}: a single input must be a matrix with {' or '.join(map(str, n))} columns.")
        return [p[:, [k]] for k in range(p.shape[1])], True
    return list(args), False


@builtin("cart2pol")
def _cart2pol(interp, args, nargout):
    nargs(args, 1, 3, "cart2pol")
    parts, single = _split_matrix(args, (2, 3), "cart2pol")
    arrs = _bcast("cart2pol", *parts)
    x, y = arrs[0], arrs[1]
    th, rho = np.arctan2(y, x), np.hypot(x, y)
    outs = [th, rho] + ([arrs[2]] if len(arrs) > 2 else [])
    if single and nargout <= 1:
        return np.hstack(outs)
    return tuple(outs[:max(nargout, 1)]) if nargout > 1 else outs[0]


@builtin("pol2cart")
def _pol2cart(interp, args, nargout):
    nargs(args, 1, 3, "pol2cart")
    parts, single = _split_matrix(args, (2, 3), "pol2cart")
    arrs = _bcast("pol2cart", *parts)
    th, rho = arrs[0], arrs[1]
    outs = [rho * np.cos(th), rho * np.sin(th)] + ([arrs[2]] if len(arrs) > 2 else [])
    if single and nargout <= 1:
        return np.hstack(outs)
    return tuple(outs[:max(nargout, 1)]) if nargout > 1 else outs[0]


@builtin("cart2sph")
def _cart2sph(interp, args, nargout):
    nargs(args, 1, 3, "cart2sph")
    parts, single = _split_matrix(args, (3,), "cart2sph")
    if len(parts) != 3:
        raise MatlabError("cart2sph: needs x, y and z.")
    x, y, z = _bcast("cart2sph", *parts)
    hxy = np.hypot(x, y)
    outs = [np.arctan2(y, x), np.arctan2(z, hxy), np.hypot(hxy, z)]
    if single and nargout <= 1:
        return np.hstack(outs)
    return tuple(outs[:max(nargout, 1)]) if nargout > 1 else outs[0]


@builtin("sph2cart")
def _sph2cart(interp, args, nargout):
    nargs(args, 1, 3, "sph2cart")
    parts, single = _split_matrix(args, (3,), "sph2cart")
    if len(parts) != 3:
        raise MatlabError("sph2cart: needs azimuth, elevation and radius.")
    az, el, r = _bcast("sph2cart", *parts)
    rc = r * np.cos(el)
    outs = [rc * np.cos(az), rc * np.sin(az), r * np.sin(el)]
    if single and nargout <= 1:
        return np.hstack(outs)
    return tuple(outs[:max(nargout, 1)]) if nargout > 1 else outs[0]


# -- special functions -----------------------------------------------------------------------

def _bessel(name, plain, scaled, needs_complex):
    def f(interp, args, nargout):
        nargs(args, 2, 3, name)
        nu, z = _bcast(name, args[0], args[1])
        if len(args) > 2 and _scalar(args[2]) not in (0, 1):
            raise MatlabError(f"{name}: the scale argument must be 0 or 1.")
        scale = len(args) > 2 and _scalar(args[2]) == 1
        fn = scaled if scale else plain
        zc = z
        if np.iscomplexobj(z) or needs_complex(nu, z):
            zc = z.astype(complex)
        with np.errstate(all="ignore"):
            r = fn(nu, zc)
        return simplify(np.asarray(r))
    return f


def _neg_noninteger(nu, z):
    return bool(np.any((np.real(z) < 0) & (nu != np.round(nu))))


def _neg_any(nu, z):
    return bool(np.any(np.real(z) < 0))


def _sp(name):
    return lazy_function("scipy.special", name)


builtin("besselj")(_bessel("besselj", _sp("jv"), _sp("jve"), _neg_noninteger))
builtin("bessely")(_bessel("bessely", _sp("yv"), _sp("yve"), _neg_any))
builtin("besseli")(_bessel("besseli", _sp("iv"), _sp("ive"), _neg_noninteger))
builtin("besselk")(_bessel("besselk", _sp("kv"), _sp("kve"), _neg_any))


@builtin("airy")
def _airy(interp, args, nargout):
    nargs(args, 1, 3, "airy")
    if len(args) == 1:
        k, z = 0, fnum(args[0])
    else:
        k, z = to_int(args[0], "k"), fnum(args[1])
    if k not in (0, 1, 2, 3):
        raise MatlabError("airy: k must be 0, 1, 2 or 3.")
    if len(args) > 2 and _scalar(args[2]) not in (0, 1):
        raise MatlabError("airy: the scale argument must be 0 or 1.")
    scaled = len(args) > 2 and _scalar(args[2]) == 1
    fn = sp.airye if scaled else sp.airy
    return simplify(np.asarray(fn(z)[k]))


@builtin("ellipke")
def _ellipke(interp, args, nargout):
    nargs(args, 1, 2, "ellipke")
    m = fnum(args[0]).astype(float)
    if len(args) > 1:
        tol = fnum(args[1])
        if tol.size != 1 or tol.dtype.kind == "c" or not tol.flat[0] > 0:
            raise MatlabError("ellipke: the tolerance must be a positive real scalar.")
        # the integrals are evaluated to full double precision, which meets any tolerance >= eps
    if np.any((m < 0) | (m > 1)):
        raise MatlabError("ellipke: m must be in the range 0 <= m <= 1.")
    K, E = sp.ellipk(m), sp.ellipe(m)
    return (K, E) if nargout > 1 else K


@builtin("expint")
def _expint(interp, args, nargout):
    nargs(args, 1, 1, "expint")
    x = fnum(args[0])
    if np.iscomplexobj(x):
        return simplify(np.asarray(sp.exp1(x)))
    if np.any(x < 0):
        # on the negative real axis E1(-t) = -Ei(t) - i*pi
        xf = x.astype(float)
        with np.errstate(all="ignore"):
            r = np.where(xf < 0, -sp.expi(-xf) - 1j * np.pi, sp.exp1(np.abs(xf)).astype(complex))
        return simplify(np.asarray(r))
    with np.errstate(divide="ignore"):
        return sp.exp1(x.astype(float))


@builtin("gammainc")
def _gammainc(interp, args, nargout):
    nargs(args, 2, 3, "gammainc")
    x, a = _bcast("gammainc", args[0], args[1])
    tail = text(args[2]).lower() if len(args) > 2 else "lower"
    if np.any(a < 0):
        raise MatlabError("gammainc: a must be non-negative.")
    if tail == "lower":
        return sp.gammainc(a, x)
    if tail == "upper":
        return sp.gammaincc(a, x)
    if tail in ("scaledlower", "scaledupper"):
        # scaled by gamma(a+1) * exp(x) / x^a, which removes the leading behaviour for large a or x
        with np.errstate(all="ignore"):
            logscale = sp.gammaln(a + 1) + x - a * np.log(x)
            base = sp.gammainc(a, x) if tail == "scaledlower" else sp.gammaincc(a, x)
            out = base * np.exp(logscale)
            if tail == "scaledlower":
                out = np.where(x == 0, 1.0, out)
            else:
                out = np.where((x == 0) & (a > 0), np.inf, out)
        return out
    raise MatlabError(f"gammainc: the tail must be 'lower', 'upper', 'scaledlower' or 'scaledupper', "
                      f"not '{tail}'.")


@builtin("betainc")
def _betainc(interp, args, nargout):
    nargs(args, 3, 4, "betainc")
    x, z, w = _bcast("betainc", args[0], args[1], args[2])
    tail = text(args[3]).lower() if len(args) > 3 else "lower"
    if np.any((x < 0) | (x > 1)):
        raise MatlabError("betainc: x must be in the interval [0, 1].")
    if tail == "lower":
        return sp.betainc(z, w, x)
    if tail == "upper":
        return sp.betaincc(z, w, x)
    raise MatlabError(f"betainc: unsupported tail '{tail}'.")


@builtin("psi")
def _psi(interp, args, nargout):
    nargs(args, 1, 2, "psi")
    if len(args) == 1:
        return sp.digamma(fnum(args[0]).astype(float))
    k, x = _bcast("psi", args[0], args[1])
    if np.any(k < 0) or np.any(k != np.round(k)):
        raise MatlabError("psi: k must be a non-negative integer.")
    return sp.polygamma(k.astype(int), x)


# -- bytes --------------------------------------------------------------------------------

_TYPECAST = {"double": np.float64, "single": np.float32, **INT_TYPES}


@builtin("typecast")
def _typecast(interp, args, nargout):
    nargs(args, 2, 2, "typecast")
    x = args[0]
    cls = text(args[1], "type name").lower()
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf" or is_char(x):
        raise MatlabError("typecast: input must be a real, non-logical numeric array.")
    if cls not in _TYPECAST:
        raise MatlabError(f"typecast: unsupported type '{cls}'.")
    if not (x.shape[0] == 1 or x.shape[1] == 1 or x.size == 0):
        raise MatlabError("typecast: input must be a scalar or vector.")
    raw = np.ascontiguousarray(x.flatten(order="F")).astype(x.dtype.newbyteorder("<")).tobytes()
    size = np.dtype(_TYPECAST[cls]).itemsize
    if len(raw) % size:
        raise MatlabError("typecast: the input size is not a multiple of the output type size.")
    out = np.frombuffer(raw, dtype=np.dtype(_TYPECAST[cls]).newbyteorder("<")).astype(_TYPECAST[cls])
    return out.reshape(-1, 1) if x.shape[1] == 1 and x.shape[0] > 1 else out.reshape(1, -1)


# -- timing ---------------------------------------------------------------------------------------

@builtin("etime")
def _etime(interp, args, nargout):
    import datetime as dt
    nargs(args, 2, 2, "etime")

    def stamp(v):
        c = fnum(v).flatten()
        if c.size != 6:
            raise MatlabError("etime expects date vectors [year month day hour minute seconds] "
                              "such as clock returns.")
        whole = int(np.floor(c[5]))
        base = dt.datetime(int(c[0]), int(c[1]), int(c[2]), int(c[3]), int(c[4]), whole)
        return base, float(c[5] - whole)

    (b1, f1), (b0, f0) = stamp(args[0]), stamp(args[1])
    return (b1 - b0).total_seconds() + (f1 - f0)


@builtin("timeit")
def _timeit(interp, args, nargout):
    import time
    nargs(args, 1, 2, "timeit")
    f = args[0]
    nout = to_int(args[1], "number of outputs") if len(args) > 1 else 0

    def once() -> float:
        t0 = time.perf_counter()
        interp.call_function_value(f, [], nout)
        return time.perf_counter() - t0

    first = once()
    runs = 3 if first > 0.25 else min(25, max(5, int(0.25 / max(first, 1e-6))))
    return float(np.median([once() for _ in range(runs)]))


# -- fourth output for fmincon (registered in optim.py) -------------------------------------------

def _with_output_struct(name: str, algorithm: str) -> None:
    from ._registry import BUILTINS
    base = BUILTINS.get(name)
    if base is None:
        return

    def wrapped(interp, args, nargout):
        if nargout < 4:
            return base(interp, args, nargout)
        fn = args[0] if args else None
        count = [0]
        patched_before = "call_function_value" in vars(interp)
        original = interp.call_function_value

        def counting(f, a, n=1):
            if f is fn:
                count[0] += 1
            return original(f, a, n)

        interp.call_function_value = counting
        try:
            res = base(interp, args, 3)
        finally:
            if patched_before:
                interp.call_function_value = original
            else:
                del interp.call_function_value
        flag = float(fnum(res[2]).flat[0])
        msg = "Local minimum found that satisfies the constraints." if flag > 0 else \
            "The solver stopped before meeting its convergence criteria."
        output = Struct.scalar({"iterations": np.nan, "funcCount": float(count[0]),
                                "algorithm": char(algorithm), "message": char(msg)})
        return (*res[:3], output)[:nargout]

    wrapped.__module__ = base.__module__
    BUILTINS[name] = wrapped


_with_output_struct("fmincon", "sqp")

