"""Array construction, size and type queries, conversions."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..ops import concat
from ..values import (
    INT_TYPES, Cell, FunctionHandle, MString, Struct, char, class_name, codes_to_char,
    is_char, saturate_int, shape_of, to_int, to_str, values_equal,
)
from ._registry import builtin
from ._util import (
    bool_scalar, cast_class, dims_from_args, fnum, is_text, nargs, num, text,
)


# -- construction ------------------------------------------------------------------

def _filled(value):
    def f(interp, args, nargout):
        shape, cls = dims_from_args(args, "array constructor")
        return cast_class(np.full(shape, value, dtype=float), cls)
    return f


builtin("zeros")(_filled(0.0))
builtin("ones")(_filled(1.0))
builtin("NaN", "nan")(_filled(np.nan))
builtin("Inf", "inf")(_filled(np.inf))


@builtin("true")
def _true(interp, args, nargout):
    shape, _ = dims_from_args(args, "true")
    return np.ones(shape, dtype=bool)


@builtin("false")
def _false(interp, args, nargout):
    shape, _ = dims_from_args(args, "false")
    return np.zeros(shape, dtype=bool)


@builtin("eye")
def _eye(interp, args, nargout):
    (r, c), cls = dims_from_args(args, "eye")
    return cast_class(np.eye(r, c), cls)


def _rng(interp):
    rng = interp.state.get("rng")
    if rng is None:
        rng = np.random.default_rng()
        interp.state["rng"] = rng
    return rng


@builtin("rand")
def _rand(interp, args, nargout):
    shape, _ = dims_from_args(args, "rand")
    return _rng(interp).random(shape)


@builtin("randn")
def _randn(interp, args, nargout):
    shape, _ = dims_from_args(args, "randn")
    return _rng(interp).standard_normal(shape)


@builtin("randi")
def _randi(interp, args, nargout):
    nargs(args, 1, None, "randi")
    lim = num(args[0]).flatten()
    lo, hi = (1, int(lim[0])) if lim.size == 1 else (int(lim[0]), int(lim[1]))
    shape, _ = dims_from_args(args[1:], "randi")
    return _rng(interp).integers(lo, hi + 1, size=shape).astype(float)


@builtin("randperm")
def _randperm(interp, args, nargout):
    n = to_int(args[0], "n")
    p = _rng(interp).permutation(n) + 1.0
    if len(args) > 1:
        p = p[: to_int(args[1], "k")]
    return p.reshape(1, -1)


@builtin("rng")
def _rngf(interp, args, nargout):
    if args and not is_text(args[0]):
        interp.state["rng"] = np.random.default_rng(to_int(args[0], "seed"))
    elif args and text(args[0]) in ("default",):
        interp.state["rng"] = np.random.default_rng(0)
    elif args and text(args[0]) == "shuffle":
        interp.state["rng"] = np.random.default_rng()
    return None


@builtin("linspace")
def _linspace(interp, args, nargout):
    nargs(args, 2, 3, "linspace")
    a, b = num(args[0]).flat[0], num(args[1]).flat[0]
    n = to_int(args[2], "n") if len(args) > 2 else 100
    return np.linspace(a, b, max(n, 0)).reshape(1, -1)


@builtin("logspace")
def _logspace(interp, args, nargout):
    a, b = float(num(args[0]).flat[0]), float(num(args[1]).flat[0])
    n = to_int(args[2], "n") if len(args) > 2 else 50
    return np.logspace(a, b, n).reshape(1, -1)


@builtin("colon")
def _colon(interp, args, nargout):
    from ..interpreter import colon_range
    if len(args) == 2:
        return colon_range(args[0], None, args[1])
    return colon_range(args[0], args[1], args[2])


@builtin("magic")
def _magic(interp, args, nargout):
    n = to_int(args[0], "n")
    if n < 1:
        return np.zeros((0, 0))
    if n == 2:
        return np.array([[4.0, 3.0], [1.0, 2.0]])
    if n % 2 == 1:
        m = np.zeros((n, n))
        i, j = 0, n // 2
        for k in range(1, n * n + 1):
            m[i, j] = k
            i2, j2 = (i - 1) % n, (j + 1) % n
            if m[i2, j2]:
                i2, j2 = (i + 1) % n, j
            i, j = i2, j2
        return m
    if n % 4 == 0:
        m = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
        idx = np.arange(n)
        band = ((idx + 1) % 4) // 2
        mask = band[:, None] == band[None, :]
        m[mask] = n * n + 1 - m[mask]
        return m
    # singly even (n = 4k + 2): build from four odd magic squares, then swap columns.
    p = n // 2
    a = _magic(interp, [np.array([[float(p)]])], 1)
    m = np.block([[a, a + 2 * p * p], [a + 3 * p * p, a + p * p]])
    k = (n - 2) // 4
    top = np.arange(p)
    cols = list(range(k)) + list(range(n - k + 1, n))
    rows_a, rows_b = np.concatenate([top, top + p]), np.concatenate([top + p, top])
    m[np.ix_(rows_a, cols)] = m[np.ix_(rows_b, cols)]
    rows_a, rows_b = np.array([k, k + p]), np.array([k + p, k])
    m[np.ix_(rows_a, [0, k])] = m[np.ix_(rows_b, [0, k])]
    return m


@builtin("repmat")
def _repmat(interp, args, nargout):
    nargs(args, 2, 3, "repmat")
    v = args[0]
    if len(args) == 2:
        r = num(args[1]).flatten()
        m, n = (int(r[0]), int(r[0])) if r.size == 1 else (int(r[0]), int(r[1]))
    else:
        m, n = to_int(args[1], "m"), to_int(args[2], "n")
    if isinstance(v, Cell):
        return Cell(np.tile(v.data, (m, n)))
    if isinstance(v, Struct):
        return Struct(np.tile(v.copy().data, (m, n)), v.fields)
    if isinstance(v, MString):
        if m * n == 1:
            return v
        raise MatlabError("String arrays are not supported yet.")
    return np.tile(v, (m, n))


@builtin("reshape")
def _reshape(interp, args, nargout):
    nargs(args, 2, None, "reshape")
    v = args[0]
    total = shape_of(v)[0] * shape_of(v)[1]
    if len(args) == 2:
        dims = [int(x) for x in num(args[1]).flatten()]
    else:
        dims = []
        unknown = None
        for k, a in enumerate(args[1:]):
            if isinstance(a, np.ndarray) and a.size == 0:
                unknown = k
                dims.append(1)
            else:
                dims.append(to_int(a, "size"))
        if unknown is not None:
            known = int(np.prod([d for i, d in enumerate(dims) if i != unknown]))
            if known == 0 or total % known:
                raise MatlabError("Size arguments must divide the number of elements.")
            dims[unknown] = total // known
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    if len(dims) != 2:
        raise MatlabError("reshape: only 2-D arrays are supported.")
    if dims[0] * dims[1] != total:
        raise MatlabError("Number of elements must not change. Use [] as one of the size inputs to "
                          "automatically calculate the appropriate size for that dimension.")
    if isinstance(v, Cell):
        return Cell(v.data.reshape(dims, order="F"))
    if isinstance(v, Struct):
        return Struct(v.data.reshape(dims, order="F"), v.fields)
    return v.reshape(dims, order="F")


@builtin("cat")
def _cat(interp, args, nargout):
    dim = to_int(args[0], "dim")
    if dim not in (1, 2):
        raise MatlabError("cat: only dimensions 1 and 2 are supported.")
    return concat(list(args[1:]), 1 if dim == 2 else 0)


@builtin("horzcat")
def _horzcat(interp, args, nargout):
    return concat(list(args), 1) if args else np.zeros((0, 0))


@builtin("vertcat")
def _vertcat(interp, args, nargout):
    return concat(list(args), 0) if args else np.zeros((0, 0))


@builtin("meshgrid")
def _meshgrid(interp, args, nargout):
    x = fnum(args[0]).flatten(order="F")
    y = fnum(args[1]).flatten(order="F") if len(args) > 1 else x
    X, Y = np.meshgrid(x, y)
    return (X, Y)


@builtin("ndgrid")
def _ndgrid(interp, args, nargout):
    x = fnum(args[0]).flatten(order="F")
    y = fnum(args[1]).flatten(order="F") if len(args) > 1 else x
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (X, Y)


@builtin("diag")
def _diag(interp, args, nargout):
    a = args[0]
    k = to_int(args[1], "k") if len(args) > 1 else 0
    if a.shape[0] == 1 or a.shape[1] == 1:
        return np.diag(a.flatten(), k)
    return np.diag(a, k).reshape(-1, 1)


# -- constants -----------------------------------------------------------------------

builtin("pi")(lambda interp, args, nargout: np.pi)
builtin("i", "j")(lambda interp, args, nargout: np.array([[1j]]))
builtin("realmax")(lambda interp, args, nargout: np.finfo(float).max)
builtin("realmin")(lambda interp, args, nargout: np.finfo(float).tiny)
builtin("flintmax")(lambda interp, args, nargout: 2.0 ** 53)
builtin("e")(lambda interp, args, nargout: np.e)


@builtin("eps")
def _eps(interp, args, nargout):
    if not args:
        return np.finfo(float).eps
    if is_text(args[0]):
        return np.finfo(np.float32 if text(args[0]) == "single" else float).eps
    x = np.abs(fnum(args[0]))
    return np.spacing(x)


@builtin("intmax")
def _intmax(interp, args, nargout):
    cls = text(args[0]) if args else "int32"
    t = INT_TYPES[cls]
    return np.array([[np.iinfo(t).max]], dtype=t)


@builtin("intmin")
def _intmin(interp, args, nargout):
    cls = text(args[0]) if args else "int32"
    t = INT_TYPES[cls]
    return np.array([[np.iinfo(t).min]], dtype=t)


# -- size & shape ------------------------------------------------------------------

@builtin("size")
def _size(interp, args, nargout):
    nargs(args, 1, 2, "size")
    r, c = shape_of(args[0])
    if len(args) == 2:
        d = to_int(args[1], "dim")
        return float(r if d == 1 else c if d == 2 else 1)
    if nargout <= 1:
        return np.array([[float(r), float(c)]])
    outs = [float(r), float(c)] + [1.0] * (nargout - 2)
    return tuple(outs[:nargout])


@builtin("numel")
def _numel(interp, args, nargout):
    r, c = shape_of(args[0])
    return float(r * c)


@builtin("length")
def _length(interp, args, nargout):
    r, c = shape_of(args[0])
    return float(0 if r == 0 or c == 0 else max(r, c))


@builtin("ndims")
def _ndims(interp, args, nargout):
    return 2.0


def _shape_pred(fn):
    def f(interp, args, nargout):
        nargs(args, 1, 1, "shape test")
        return bool_scalar(fn(*shape_of(args[0])))
    return f


builtin("isempty")(_shape_pred(lambda r, c: r == 0 or c == 0))
builtin("isscalar")(_shape_pred(lambda r, c: r == 1 and c == 1))
builtin("isvector")(_shape_pred(lambda r, c: (r == 1 or c == 1) and r * c >= 1))
builtin("isrow")(_shape_pred(lambda r, c: r == 1))
builtin("iscolumn")(_shape_pred(lambda r, c: c == 1))
builtin("ismatrix")(_shape_pred(lambda r, c: True))
builtin("issquare")(_shape_pred(lambda r, c: r == c))


# -- type queries ------------------------------------------------------------------

@builtin("class")
def _class(interp, args, nargout):
    return char(class_name(args[0]))


def _type_pred(fn):
    def f(interp, args, nargout):
        nargs(args, 1, 1, "type test")
        return bool_scalar(fn(args[0]))
    return f


builtin("isnumeric")(_type_pred(lambda v: isinstance(v, np.ndarray) and v.dtype.kind in "iufc"))
builtin("isfloat")(_type_pred(lambda v: isinstance(v, np.ndarray) and v.dtype.kind in "fc"))
builtin("isinteger")(_type_pred(lambda v: isinstance(v, np.ndarray) and v.dtype.kind in "iu"))
builtin("islogical")(_type_pred(lambda v: isinstance(v, np.ndarray) and v.dtype == bool))
builtin("ischar")(_type_pred(is_char))
builtin("isstring")(_type_pred(lambda v: isinstance(v, MString)))
builtin("iscell")(_type_pred(lambda v: isinstance(v, Cell)))
builtin("isstruct")(_type_pred(lambda v: isinstance(v, Struct)))
builtin("isreal")(_type_pred(lambda v: not (isinstance(v, np.ndarray) and v.dtype.kind == "c")))
builtin("is_function_handle")(_type_pred(lambda v: isinstance(v, FunctionHandle)))
builtin("istext")(_type_pred(lambda v: isinstance(v, MString) or is_char(v)))


@builtin("isa")
def _isa(interp, args, nargout):
    cls = class_name(args[0])
    want = text(args[1])
    if want == "numeric":
        return bool_scalar(cls in ("double", "single") or cls in INT_TYPES)
    if want == "float":
        return bool_scalar(cls in ("double", "single"))
    if want == "integer":
        return bool_scalar(cls in INT_TYPES)
    return bool_scalar(cls == want)


@builtin("isequal")
def _isequal(interp, args, nargout):
    nargs(args, 2, None, "isequal")
    return bool_scalar(all(values_equal(args[0], b) for b in args[1:]))


@builtin("isequaln")
def _isequaln(interp, args, nargout):
    def norm(v):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            v = v.copy()
            v[np.isnan(v)] = -1.2345e300
        return v
    return bool_scalar(all(values_equal(norm(args[0]), norm(b)) for b in args[1:]))


# -- conversions ----------------------------------------------------------------------

@builtin("double")
def _double(interp, args, nargout):
    v = args[0]
    if isinstance(v, MString):
        try:
            return float(str(v))
        except ValueError:
            return np.nan
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            return num(v)
        return v.astype(complex if v.dtype.kind == "c" else float)
    raise MatlabError(f"Conversion to double from {class_name(v)} is not possible.")


@builtin("single")
def _single(interp, args, nargout):
    return num(args[0]).astype(np.float32 if num(args[0]).dtype.kind != "c" else np.complex64)


@builtin("logical")
def _logical(interp, args, nargout):
    a = num(args[0])
    if a.dtype.kind == "f" and np.isnan(a).any():
        raise MatlabError("NaN's cannot be converted to logicals.")
    return a != 0


def _int_conv(name):
    def f(interp, args, nargout):
        return saturate_int(num(args[0]), INT_TYPES[name])
    return f


for _n in INT_TYPES:
    builtin(_n)(_int_conv(_n))


@builtin("char")
def _char(interp, args, nargout):
    if not args:
        return char("")
    if len(args) == 1:
        v = args[0]
        if isinstance(v, MString):
            return char(str(v))
        if isinstance(v, Cell):
            return _char_rows([to_str(x) for x in v.values_f()])
        if isinstance(v, np.ndarray):
            return codes_to_char(v)
    return _char_rows([to_str(a) if is_text(a) else to_str(codes_to_char(a)) for a in args])


def _char_rows(rows: list[str]) -> np.ndarray:
    if not rows:
        return char("")
    width = max(len(r) for r in rows)
    return np.array([list(r.ljust(width)) for r in rows], dtype="<U1").reshape(len(rows), width)


@builtin("cast")
def _cast(interp, args, nargout):
    return cast_class(num(args[0]), text(args[1]))


@builtin("typecast")
def _typecast(interp, args, nargout):
    raise MatlabError("typecast is not supported yet.")


@builtin("full", "sparse")
def _full(interp, args, nargout):
    if len(args) == 1:
        return args[0]
    if len(args) >= 2 and all(not is_text(a) for a in args[:2]) and len(args) == 2:
        return np.zeros((to_int(args[0]), to_int(args[1])))
    i = fnum(args[0]).flatten().astype(int) - 1
    j = fnum(args[1]).flatten().astype(int) - 1
    v = fnum(args[2]).flatten()
    m = to_int(args[3]) if len(args) > 3 else int(i.max()) + 1
    n = to_int(args[4]) if len(args) > 4 else int(j.max()) + 1
    out = np.zeros((m, n), dtype=v.dtype)
    v = np.broadcast_to(v, i.shape)
    np.add.at(out, (i, j), v)
    return out


@builtin("issparse")
def _issparse(interp, args, nargout):
    return bool_scalar(False)

