"""Cell arrays, structs and functional helpers (cellfun, arrayfun)."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..ops import build_matrix
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, as_2d, char, class_name, empty_double, is_char, to_str,
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
    if c.data.ndim > 2:
        raise MatlabError("cell2mat: cell arrays with more than two dimensions are not supported.")
    kinds = {_content_class(x) for x in c.data.flat}
    if len(kinds) > 1:
        raise MatlabError("cell2mat: all contents of the input cell array must be of the same data type.",
                          "MATLAB:cell2mat:MixedDataTypes")
    if kinds & {"cell", "function_handle", "other"}:
        raise MatlabError("cell2mat: cannot support cell arrays containing cell arrays or objects.",
                          "MATLAB:cell2mat:UnsupportedCellArrayContents")
    rows = [[c.data[i, j] for j in range(c.shape[1])] for i in range(c.shape[0])]
    return build_matrix(rows)


def _content_class(x) -> str:
    if isinstance(x, np.ndarray):
        if x.dtype.kind == "U":
            return "char"
        if x.dtype.kind == "c":
            return "single" if x.dtype == np.complex64 else "double"
        return {"bool": "logical", "float64": "double", "float32": "single"}.get(x.dtype.name, x.dtype.name)
    if isinstance(x, (MString, StringArray)):
        return "string"
    if isinstance(x, Cell):
        return "cell"
    if isinstance(x, Struct):
        return "struct"
    if isinstance(x, FunctionHandle):
        return "function_handle"
    return "other"


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
    if not isinstance(v, np.ndarray):
        inner = np.empty((1, 1), dtype=object)
        inner[0, 0] = v
        return Cell(inner)
    if len(args) > 1:
        # num2cell(A, dims): the listed dimensions stay together inside each cell
        dims = sorted({int(d) - 1 for d in num(args[1]).flatten()})
        if any(d < 0 for d in dims):
            raise MatlabError("num2cell: dimensions must be positive integers.")
        nd = max(v.ndim, max(dims) + 1 if dims else 0)
        a = v.reshape(v.shape + (1,) * (nd - v.ndim))
        out_shape = tuple(1 if d in dims else a.shape[d] for d in range(nd))
        data = np.empty(out_shape, dtype=object)
        for idx in np.ndindex(out_shape):
            sel = tuple(slice(None) if d in dims else slice(idx[d], idx[d] + 1) for d in range(nd))
            data[idx] = as_2d(a[sel].copy())
        while data.ndim > 2 and data.shape[-1] == 1:
            data = data.reshape(data.shape[:-1])
        return Cell(data)
    data = np.empty(v.shape, dtype=object)
    for idx in np.ndindex(v.shape):
        data[idx] = v[idx].reshape(1, 1).copy()
    return Cell(data)


@builtin("struct")
def _struct(interp, args, nargout):
    if not args:
        return Struct()
    if len(args) == 1:
        v = args[0]
        if isinstance(v, Struct):
            return v
        if isinstance(v, np.ndarray) and v.size == 0:
            return Struct(np.empty((0, 0), dtype=object), [])  # struct([]): 0x0 struct without fields
        if hasattr(v, "to_struct"):
            return v.to_struct()
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


def _need_struct(v, name: str) -> Struct:
    if not isinstance(v, Struct):
        raise MatlabError(f"{name}: the input must be a struct.")
    return v


@builtin("numfields")
def _numfields(interp, args, nargout):
    nargs(args, 1, 1, "numfields")
    v = args[0]
    return np.array([[float(len(v.fields)) if isinstance(v, Struct) else 0.0]])


@builtin("struct2cell")
def _struct2cell(interp, args, nargout):
    nargs(args, 1, 1, "struct2cell")
    s = _need_struct(args[0], "struct2cell")
    if s.data.size == 1:
        return Cell.from_list([s.data.flat[0][f] for f in s.fields], column=True)
    # a struct array gives one layer of field values per element: fields x size(s)
    shape = (len(s.fields),) + tuple(s.data.shape)
    data = np.empty(shape, dtype=object)
    for idx in np.ndindex(s.data.shape):
        for k, f in enumerate(s.fields):
            data[(k,) + idx] = s.data[idx][f]
    while data.ndim > 2 and data.shape[-1] == 1:
        data = data.reshape(data.shape[:-1])
    return Cell(data)


@builtin("cell2struct")
def _cell2struct(interp, args, nargout):
    nargs(args, 2, 3, "cell2struct")
    c, f = args[0], _as_cell(args[1])
    if not isinstance(c, Cell):
        raise MatlabError("cell2struct: the first input must be a cell array.")
    if isinstance(f, Cell):
        names = [to_str(x) for x in f.values_f()]
    elif is_char(f) and f.ndim == 2 and f.shape[0] > 1:
        names = ["".join(r).rstrip() for r in f]
    else:
        names = [text(f)]
    if len(set(names)) != len(names):
        raise MatlabError("cell2struct: field names must be unique.")
    dim = int(num(args[2]).flat[0]) - 1 if len(args) > 2 else 0
    if dim < 0:
        raise MatlabError("cell2struct: the dimension must be a positive integer.")
    data = c.data
    if data.ndim <= dim:
        data = data.reshape(data.shape + (1,) * (dim + 1 - data.ndim))
    if data.shape[dim] != len(names):
        raise MatlabError("Number of field names must match number of fields in new structure.",
                          "MATLAB:cell2struct:FieldMismatch")
    moved = np.moveaxis(data, dim, 0)
    rest = moved.shape[1:]
    out_shape = rest if len(rest) >= 2 else (rest + (1, 1))[:2]
    out = np.empty(out_shape, dtype=object)
    flat = moved.reshape((len(names), -1), order="F")
    cells = np.empty(flat.shape[1], dtype=object)
    for k in range(flat.shape[1]):
        cells[k] = {name: flat[i, k] for i, name in enumerate(names)}
    out[...] = cells.reshape(out_shape, order="F") if cells.size else out
    return Struct(out, names)


def _as_cell(v):
    if isinstance(v, StringArray):
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            data[idx] = char(v.data[idx] or "")
        return Cell(data)
    return v


@builtin("orderfields")
def _orderfields(interp, args, nargout):
    nargs(args, 1, 2, "orderfields")
    src = _need_struct(args[0], "orderfields")
    fields = list(src.fields)
    if len(args) == 1:
        order = sorted(fields)
    else:
        ref = _as_cell(args[1])
        if isinstance(ref, Struct):
            order = list(ref.fields)
        elif isinstance(ref, Cell):
            order = [to_str(x) for x in ref.values_f()]
        elif isinstance(ref, np.ndarray) and not is_char(ref):
            perm = [int(x) for x in num(ref).flatten(order="F")]
            if sorted(perm) != list(range(1, len(fields) + 1)):
                raise MatlabError("orderfields: the permutation vector must list every field exactly once.")
            order = [fields[k - 1] for k in perm]
        else:
            raise MatlabError("orderfields: the second input must be a struct, a cell array of names "
                              "or a permutation vector.")
        if sorted(order) != sorted(fields):
            raise MatlabError("orderfields: both inputs must have the same field names.")
    s = src.copy()
    s.fields = order
    for d in s.data.flat:
        ordered = {name: d[name] for name in order}
        d.clear()
        d.update(ordered)
    if nargout > 1:
        perm = np.array([fields.index(name) + 1.0 for name in order]).reshape(-1, 1)
        return s, (perm if order else np.zeros((0, 1)))
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
    count = int(np.prod(shape))
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
            final.append(_uniform_output(vals, shape, i))
        else:
            data = np.empty(count, dtype=object)
            for k, v in enumerate(vals):
                data[k] = v
            final.append(Cell(data.reshape(shape, order="F")))
    return tuple(final) if len(final) > 1 else final[0]


def _uniform_output(vals: list, shape, out_index: int):
    """Join scalar results into one array that keeps their class."""
    if vals and all(isinstance(v, MString) for v in vals):
        from ..values import string_result
        data = np.empty(len(vals), dtype=object)
        for k, v in enumerate(vals):
            data[k] = str(v)
        return string_result(data.reshape(shape, order="F"))
    for k, v in enumerate(vals):
        if not isinstance(v, np.ndarray) or v.size != 1:
            raise MatlabError(f"Non-scalar in Uniform output, at index {k + 1}, output {out_index + 1}. "
                              "Set 'UniformOutput' to false.")
    if not vals:
        return np.zeros(shape)
    dtypes = {v.dtype for v in vals}
    if len(dtypes) == 1:
        dtype = vals[0].dtype  # every result has the same class: keep it (int8 stays int8)
    elif all(d.kind in "bfc" for d in dtypes):
        dtype = np.dtype(complex) if any(d.kind == "c" for d in dtypes) else np.dtype(float)
    else:
        raise MatlabError(f"Mismatch in type of outputs, at index 2, output {out_index + 1}. "
                          "Set 'UniformOutput' to false.")
    arr = np.empty(len(vals), dtype=dtype)
    for k, v in enumerate(vals):
        arr[k] = v.flat[0]
    return arr.reshape(shape, order="F")


def _legacy_cellfun(name, c, extra):
    """The old text names of cellfun ('isempty', 'prodofsize', 'size', 'isclass', ...)."""
    def shape_of(v):
        if isinstance(v, (Cell, Struct, StringArray)):
            return v.data.shape
        return v.shape if isinstance(v, np.ndarray) else (1, 1)

    def size_k(v):
        k = int(num(extra[0]).flat[0])
        sh = shape_of(v)
        return sh[k - 1] if 1 <= k <= len(sh) else 1

    legacy = {
        "isempty": lambda v: int(np.prod(shape_of(v))) == 0,
        "length": lambda v: 0 if 0 in shape_of(v) else max(shape_of(v)),
        "numel": lambda v: int(np.prod(shape_of(v))),
        "prodofsize": lambda v: int(np.prod(shape_of(v))),
        "ndims": lambda v: len(shape_of(v)),
        "isreal": lambda v: isinstance(v, np.ndarray) and v.dtype.kind != "c",
        "islogical": lambda v: isinstance(v, np.ndarray) and v.dtype == bool,
        "size": size_k,
        "isclass": lambda v: class_name(v) == text(extra[0]),
    }
    if name not in legacy:
        return None
    if len(extra) != (1 if name in ("size", "isclass") else 0):
        if name in ("size", "isclass"):
            raise MatlabError(f"cellfun: '{name}' needs one more argument.")
        return None
    fn = legacy[name]
    out = np.array([fn(x) for x in c.data.flatten(order="F")]).reshape(c.shape, order="F")
    return out.astype(bool if name in ("isempty", "isreal", "islogical", "isclass") else float)


@builtin("cellfun")
def _cellfun(interp, args, nargout):
    nargs(args, 2, None, "cellfun")
    rest, opts = _split_options(args, ("UniformOutput", "ErrorHandler"))
    fn = rest[0]
    cells = rest[1:]
    if is_text(fn) and cells and isinstance(cells[0], Cell):
        res = _legacy_cellfun(text(fn), cells[0], cells[1:])
        if res is not None:
            return res
    for c in cells:
        if not isinstance(c, Cell):
            raise MatlabError("cellfun: all inputs after the function must be cell arrays.")
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
