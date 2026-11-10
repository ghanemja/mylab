"""Cell arrays, structs and functional helpers (cellfun, arrayfun)."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..ops import build_matrix
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, empty_double, is_char, to_str,
)
from ._registry import builtin
from ._util import bool_scalar, dims_from_args, is_text, nargs, num, text


@builtin("cell")
def _cell(interp, args, nargout):
    shape, _ = dims_from_args(args, "cell")
    data = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        data[idx] = np.zeros((0, 0))
    return Cell(data)


@builtin("iscellstr")
def _iscellstr(interp, args, nargout):
    v = args[0]
    return bool_scalar(isinstance(v, Cell) and all(is_char(x) and x.shape[0] <= 1 for x in v.data.flat))


@builtin("cellstr")
def _cellstr(interp, args, nargout):
    v = args[0]
    if isinstance(v, Cell):
        return v
    if isinstance(v, MString):
        return Cell.from_list([char(str(v))])
    if isinstance(v, StringArray):
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            data[idx] = char(v.data[idx] or "")
        return Cell(data)
    if is_char(v):
        rows = ["".join(r).rstrip() for r in v] if v.size else [""]
        return Cell.from_list([char(r) for r in rows], column=True)
    raise MatlabError("cellstr: input must be text.")


@builtin("cell2mat")
def _cell2mat(interp, args, nargout):
    c = args[0]
    if not isinstance(c, Cell):
        raise MatlabError("cell2mat: input must be a cell array.")
    if c.data.size == 0:
        return np.zeros((0, 0))
    rows = [[c.data[i, j] for j in range(c.shape[1])] for i in range(c.shape[0])]
    return build_matrix(rows)


@builtin("num2cell")
def _num2cell(interp, args, nargout):
    v = args[0]
    if isinstance(v, Cell):
        data = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            inner = np.empty((1, 1), dtype=object)
            inner[0, 0] = v.data[idx]
            data[idx] = Cell(inner)
        return Cell(data)
    data = np.empty(v.shape, dtype=object)
    for idx in np.ndindex(v.shape):
        data[idx] = v[idx[0]:idx[0] + 1, idx[1]:idx[1] + 1].copy()
    return Cell(data)


@builtin("struct")
def _struct(interp, args, nargout):
    if not args:
        return Struct()
    if len(args) % 2:
        raise MatlabError("struct: field names and values must come in pairs.")
    names = [text(args[k], "field name") for k in range(0, len(args), 2)]
    values = [args[k] for k in range(1, len(args), 2)]
    # Cell values create struct arrays; 1x1 cells are unwrapped.
    shape = (1, 1)
    for v in values:
        if isinstance(v, Cell) and v.data.size != 1:
            shape = v.shape
            break
    data = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        elem = {}
        for name, v in zip(names, values):
            if isinstance(v, Cell):
                if v.data.size == 1:
                    elem[name] = v.data.flat[0]
                elif v.data.size == 0:
                    elem[name] = empty_double()
                else:
                    if v.shape != shape:
                        raise MatlabError("struct: cell array values must all be the same size.")
                    elem[name] = v.data[idx]
            else:
                elem[name] = v
        data[idx] = elem
    return Struct(data, names)


@builtin("fieldnames")
def _fieldnames(interp, args, nargout):
    s = args[0]
    if not isinstance(s, Struct):
        raise MatlabError("fieldnames: input must be a struct.")
    return Cell.from_list([char(f) for f in s.fields], column=True)


@builtin("isfield")
def _isfield(interp, args, nargout):
    s, f = args[0], args[1]
    if not isinstance(s, Struct):
        return bool_scalar(False)
    if isinstance(f, Cell):
        return np.vectorize(lambda x: to_str(x) in s.fields, otypes=[bool])(f.data)
    return bool_scalar(text(f) in s.fields)


@builtin("rmfield")
def _rmfield(interp, args, nargout):
    s = args[0].copy()
    names = [to_str(x) for x in args[1].values_f()] if isinstance(args[1], Cell) else [text(args[1])]
    for n in names:
        if n not in s.fields:
            raise MatlabError(f"A field named '{n}' doesn't exist.")
        s.fields.remove(n)
        for d in s.data.flat:
            d.pop(n, None)
    return s


@builtin("setfield")
def _setfield(interp, args, nargout):
    s = args[0].copy() if isinstance(args[0], Struct) else Struct()
    s.set_field(0, text(args[1]), args[2])
    return s


@builtin("getfield")
def _getfield(interp, args, nargout):
    return args[0].get_field(0, text(args[1]))


@builtin("struct2cell")
def _struct2cell(interp, args, nargout):
    s = args[0]
    return Cell.from_list([s.data.flat[0][f] for f in s.fields], column=True)


@builtin("cell2struct")
def _cell2struct(interp, args, nargout):
    c, f = args[0], args[1]
    names = [to_str(x) for x in f.values_f()] if isinstance(f, Cell) else [text(f)]
    vals = c.values_f()
    if len(vals) != len(names):
        raise MatlabError("cell2struct: number of fields must match number of values.")
    return Struct.scalar(dict(zip(names, vals)))


@builtin("orderfields")
def _orderfields(interp, args, nargout):
    s = args[0].copy()
    s.fields = sorted(s.fields)
    return s


@builtin("deal")
def _deal(interp, args, nargout):
    n = max(nargout, 1)
    if len(args) == 1:
        return tuple([args[0]] * n) if n > 1 else args[0]
    if len(args) != n:
        raise MatlabError("The number of outputs should match the number of inputs.")
    return tuple(args) if n > 1 else args[0]


def _split_options(args, names):
    opts = {}
    rest = list(args)
    while len(rest) >= 2 and is_text(rest[-2]) and text(rest[-2]) in names:
        opts[text(rest[-2])] = rest[-1]
        rest = rest[:-2]
    return rest, opts


def _apply_elementwise(interp, fn, containers, nargout, uniform, error_handler, shape):
    n_out = max(nargout, 1)
    count = shape[0] * shape[1]
    results = [[None] * count for _ in range(n_out)]
    for k in range(count):
        call_args = [c[k] for c in containers]
        try:
            outs = interp.call_function_value(fn, call_args, n_out if nargout > 0 else 1)
        except MatlabError as e:
            if error_handler is None:
                raise
            err = Struct.scalar({"message": char(e.message), "identifier": char(e.identifier),
                                 "index": np.array([[k + 1.0]])})
            outs = interp.call_function_value(error_handler, [err, *call_args], n_out)
        if len(outs) < n_out:
            if nargout == 0 and not outs:
                n_out = 0
                results = []
                continue
            raise MatlabError("The function returned fewer outputs than requested.")
        for i in range(n_out):
            results[i][k] = outs[i]
    if n_out == 0:
        return None
    final = []
    for i in range(n_out):
        vals = results[i]
        if uniform:
            arr = np.zeros(shape)
            kind = None
            for k, v in enumerate(vals):
                if not isinstance(v, np.ndarray) or v.size != 1:
                    raise MatlabError("Non-scalar in Uniform output, at index "
                                      f"{k + 1}, output {i + 1}. Set 'UniformOutput' to false.")
                if v.dtype == bool and kind in (None, bool):
                    kind = bool
                elif v.dtype.kind == "c":
                    kind = complex
                    arr = arr.astype(complex)
                elif v.dtype.kind == "U":
                    kind = "U"
                else:
                    kind = float if kind is not complex else complex
                arr[k % shape[0], k // shape[0]] = v.flat[0] if v.dtype.kind != "U" else ord(v.flat[0])
            if kind is bool:
                arr = arr.astype(bool)
            elif kind == "U":
                from ..values import codes_to_char
                arr = codes_to_char(arr)
            final.append(arr)
        else:
            data = np.empty(shape, dtype=object)
            for k, v in enumerate(vals):
                data[k % shape[0], k // shape[0]] = v
            final.append(Cell(data))
    return tuple(final) if len(final) > 1 else final[0]


def _legacy_cellfun(name, c):
    legacy = {
        "isempty": lambda v: np.size(getattr(v, "data", v)) == 0 if not isinstance(v, str) else False,
        "length": lambda v: 0 if 0 in np.shape(getattr(v, "data", v)) else max(np.shape(getattr(v, "data", v))),
        "numel": lambda v: int(np.prod(np.shape(getattr(v, "data", v)))),
        "isreal": lambda v: not (isinstance(v, np.ndarray) and v.dtype.kind == "c"),
        "islogical": lambda v: isinstance(v, np.ndarray) and v.dtype == bool,
        "ndims": lambda v: 2,
    }
    if name in legacy:
        fn = legacy[name]
        out = np.array([fn(x) for x in c.data.flatten(order="F")]).reshape(c.shape, order="F")
        return out.astype(bool if name in ("isempty", "isreal", "islogical") else float)
    return None


@builtin("cellfun")
def _cellfun(interp, args, nargout):
    nargs(args, 2, None, "cellfun")
    rest, opts = _split_options(args, ("UniformOutput", "ErrorHandler"))
    fn = rest[0]
    cells = rest[1:]
    for c in cells:
        if not isinstance(c, Cell):
            raise MatlabError("cellfun: all inputs after the function must be cell arrays.")
    if is_text(fn) and len(cells) == 1:
        res = _legacy_cellfun(text(fn), cells[0])
        if res is not None:
            return res
    shape = cells[0].shape
    for c in cells[1:]:
        if c.shape != shape:
            raise MatlabError("All of the input arguments must be of the same size and shape.")
    uniform = bool(num(opts["UniformOutput"]).flat[0]) if "UniformOutput" in opts else True
    containers = [c.data.flatten(order="F") for c in cells]
    return _apply_elementwise(interp, fn, containers, nargout, uniform, opts.get("ErrorHandler"), shape)


def _elements(v):
    """Split an array value into per-element values (column-major)."""
    if isinstance(v, Cell):
        out = []
        for x in v.data.flatten(order="F"):
            d = np.empty((1, 1), dtype=object)
            d[0, 0] = x
            out.append(Cell(d))
        return out, v.shape
    if isinstance(v, Struct):
        out = []
        for d in v.data.flatten(order="F"):
            arr = np.empty((1, 1), dtype=object)
            arr[0, 0] = dict(d)
            out.append(Struct(arr, v.fields))
        return out, v.shape
    if isinstance(v, (MString, FunctionHandle)):
        return [v], (1, 1)
    flat = v.flatten(order="F")
    return [flat[k:k + 1].reshape(1, 1) for k in range(flat.size)], v.shape


@builtin("arrayfun")
def _arrayfun(interp, args, nargout):
    nargs(args, 2, None, "arrayfun")
    rest, opts = _split_options(args, ("UniformOutput", "ErrorHandler"))
    fn = rest[0]
    split = [_elements(a) for a in rest[1:]]
    shape = split[0][1]
    for _, s in split[1:]:
        if s != shape:
            raise MatlabError("All of the input arguments must be of the same size and shape.")
    uniform = bool(num(opts["UniformOutput"]).flat[0]) if "UniformOutput" in opts else True
    containers = [vals for vals, _ in split]
    return _apply_elementwise(interp, fn, containers, nargout, uniform, opts.get("ErrorHandler"), shape)


@builtin("structfun")
def _structfun(interp, args, nargout):
    rest, opts = _split_options(args, ("UniformOutput", "ErrorHandler"))
    fn, s = rest[0], rest[1]
    vals = [s.data.flat[0][f] for f in s.fields]
    uniform = bool(num(opts["UniformOutput"]).flat[0]) if "UniformOutput" in opts else True
    res = _apply_elementwise(interp, fn, [vals], nargout, uniform, None, (len(vals), 1))
    if not uniform and isinstance(res, Cell):
        return Struct.scalar(dict(zip(s.fields, res.values_f())))
    return res


@builtin("containers.Map")
def _containers_map(interp, args, nargout):
    from ..objects import make_map
    return make_map(list(args))


@builtin("isobject")
def _isobject(interp, args, nargout):
    from ..objects import MObject
    return bool_scalar(isinstance(args[0], MObject))
