"""More table functions: reshaping (stack, unstack, pivot, rows2vars, mergevars, splitvars),
functions applied by row, variable or group (rowfun, varfun, groupcounts, groupfilter,
grouptransform, groupsummary with bins), set operations and missing data on tables,
variable selection by type (vartype), summary, and discretize.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, class_name, is_char, string_result, to_numeric,
    to_str,
)
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, is_text, nargs, text
from .table import (
    _MISSING, _TABLE_FUNCS, TableObject, _as_column, _col_keys, _group_rows, _height, _is_numeric_var, _is_time,
    _method_list, _missing_mask, _names_from, _rows, _sort_key, _stack_results,
)


def _need_table(args: list, name: str, pos: int = 0) -> TableObject:
    if len(args) <= pos or not isinstance(args[pos], TableObject):
        raise MatlabError(f"{name}: the {'first' if pos == 0 else 'second'} argument must be a table.",
                          "MATLAB:table:InvalidInput")
    return args[pos]


def _options(args: list, names: tuple, name: str, start: int = 0) -> tuple[list, dict]:
    """Split trailing Name, Value pairs (names are not case sensitive) from the leading arguments."""
    lookup = {n.lower(): n for n in names}
    rest = list(args)
    opts: dict = {}
    while len(rest) - start >= 2 and is_text(rest[-2]) and text(rest[-2]).lower() in lookup:
        opts[lookup[text(rest[-2]).lower()]] = rest[-1]
        del rest[-2:]
    return rest, opts


def _flag(v: Any) -> bool:
    if is_text(v):
        return text(v).lower() in ("true", "on", "yes")
    return bool(np.real(to_numeric(v)).flat[0])


def _var_names(t: TableObject, spec: Any) -> list[str]:
    """Variable names selected by names, numbers, a logical mask, a vartype or a function handle."""
    if isinstance(spec, FunctionHandle):
        from ..interpreter import current_interpreter
        interp = current_interpreter()
        out = []
        for n, c in zip(t.names, t.columns):
            res = interp.call_function_value(spec, [c], 1)
            if res and bool(np.all(to_numeric(res[0]))):
                out.append(n)
        return out
    return [t.names[p] for p in t.var_positions(spec)]


def _valid_name(s: str) -> str:
    """A valid variable name made from any text (as matlab.lang.makeValidName does)."""
    out = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if not out or not (out[0].isalpha()):
        out = "x" + out
    return out


def _key_text(k: Any) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, bool):
        return "true" if k else "false"
    if isinstance(k, float):
        return str(int(k)) if k == int(k) and abs(k) < 1e15 else f"{k:.5g}"
    return str(k)


def _column_from_values(values: list) -> Any:
    """A table variable from one value per row (numbers stay numeric, text becomes a cellstr ...)."""
    if values and all(isinstance(v, np.ndarray) and not is_char(v) and v.ndim == 2 and v.shape[0] == 1
                      for v in values) and len({v.shape[1] for v in values}) == 1 \
            and len({v.dtype.kind in "iu" and v.dtype.name or v.dtype.kind for v in values}) == 1:
        return np.vstack(values)
    if values and all(isinstance(v, MString) for v in values):
        data = np.empty((len(values), 1), dtype=object)
        for i, v in enumerate(values):
            data[i, 0] = str(v)
        return StringArray(data)
    if values and all(_is_time(v) and v.data.size == 1 for v in values) and len({type(v) for v in values}) == 1:
        return values[0].like(np.array([[float(v.data.flat[0])] for v in values]))
    data = np.empty((len(values), 1), dtype=object)
    for i, v in enumerate(values):
        data[i, 0] = v
    return Cell(data)


def _row_value(col: Any, i: int) -> Any:
    """The value of one row of a table variable, as a function argument."""
    if isinstance(col, Cell):
        d = np.empty((1,) + col.data.shape[1:], dtype=object)
        d[0] = col.data[i]
        return Cell(d)
    if isinstance(col, StringArray):
        return string_result(col.data[i:i + 1])
    return _rows(col, np.array([i]))


# -- vartype ---------------------------------------------------------------------------------------------

class VartypeSubscript:
    """T(:, vartype('numeric')) selects the variables of a type."""

    is_vartype = True

    def __init__(self, kind: str):
        self.kind = kind

    def isa_class(self, name: str) -> bool:
        return name in ("vartype", "matlab.internal.tabular.private.vartypeIndexer")

    def class_name(self) -> str:
        return "vartype"

    def matlab_shape(self) -> tuple:
        return (1, 1)

    def has_method(self, name: str) -> bool:
        return False

    def matches(self, col: Any) -> bool:
        k = self.kind
        numeric = isinstance(col, np.ndarray) and col.dtype.kind in "iufc"
        if k == "numeric":
            return numeric
        if k == "float":
            return isinstance(col, np.ndarray) and col.dtype.kind in "fc"
        if k == "integer":
            return isinstance(col, np.ndarray) and col.dtype.kind in "iu"
        if k == "cellstr":
            return isinstance(col, Cell) and all(is_char(x) for x in col.data.flat)
        if k == "char":
            return is_char(col)
        if k in ("table", "tabular"):
            return isinstance(col, TableObject)
        have = class_name(col)
        return have == k

    def display_lines(self) -> list[str]:
        return ["  table vartype subscript:", "", f"    Select table variables matching the type '{self.kind}'"]

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


@builtin("vartype")
def _vartype(interp, args, nargout):
    """vartype(type): a subscript that selects the table variables of that type, e.g. T(:, vartype('numeric'))."""
    nargs(args, 1, 1, "vartype")
    return VartypeSubscript(text(args[0], "type name"))


# -- discretize ------------------------------------------------------------------------------------------------

def _edge_text(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.5g}"


@builtin("discretize")
def _discretize(interp, args, nargout):
    """discretize(X, edges | N[, values | 'categorical'[, names]], 'IncludedEdge', side)."""
    nargs(args, 2, None, "discretize")
    rest, opts = _options(list(args), ("IncludedEdge",), "discretize", 2)
    right = "IncludedEdge" in opts and text(opts["IncludedEdge"]).lower() == "right"
    if "IncludedEdge" in opts and text(opts["IncludedEdge"]).lower() not in ("left", "right"):
        raise MatlabError("discretize: 'IncludedEdge' must be 'left' or 'right'.", "MATLAB:discretize:InvalidSide")
    x = rest[0]
    is_time = _is_time(x) and getattr(x, "kind", "") in ("datetime", "duration")
    xv = np.asarray(x.data if is_time else np.real(to_numeric(x)), dtype=float)
    e = rest[1]
    ev = np.asarray(e.data if _is_time(e) else np.real(to_numeric(e)), dtype=float).flatten(order="F")
    if ev.size == 1 and not _is_time(e):
        nb = int(ev[0])
        if nb < 1 or nb != ev[0]:
            raise MatlabError("discretize: the number of bins must be a positive whole number.",
                              "MATLAB:discretize:InvalidBins")
        finite = xv[np.isfinite(xv)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        # the same "nice" edges that histcounts picks for a number of bins
        raw = (hi - lo) / nb
        power = 10.0 ** np.floor(np.log10(raw))
        frac = raw / power
        step = power * (1 if frac <= 1 else 2 if frac <= 2 else 5 if frac <= 5 else 10)
        start = np.floor(lo / step) * step
        ev = start + step * np.arange(nb + 1)
        if ev[-1] < hi:
            ev = np.linspace(lo, hi, nb + 1)
    if ev.size < 2 or np.any(np.diff(ev) < 0):
        raise MatlabError("discretize: the edges must be a vector of at least two increasing values.",
                          "MATLAB:discretize:InvalidEdges")
    nbins = ev.size - 1
    with np.errstate(invalid="ignore"):
        if right:
            idx = np.searchsorted(ev, xv, side="left").astype(float)
            idx = np.where(xv == ev[0], 1.0, idx)
        else:
            idx = np.searchsorted(ev, xv, side="right").astype(float)
            idx = np.where(xv == ev[-1], float(nbins), idx)
        idx = np.where((xv < ev[0]) | (xv > ev[-1]) | np.isnan(xv), np.nan, idx)
    edges_out = e if _is_time(e) else ev.reshape(1, -1)
    result: Any = idx
    if len(rest) > 2:
        third = rest[2]
        if is_text(third) and text(third).lower() == "categorical":
            from .categorical import CategoricalArray
            if len(rest) > 3:
                names = _names_from(rest[3])
                if len(names) != nbins:
                    raise MatlabError("discretize: give one category name for each bin.",
                                      "MATLAB:discretize:CategoryNamesInvalidSize")
            else:
                if _is_time(e):
                    labels = [str(s) for s in np.asarray(e.element_texts(), dtype=object).flatten(order="F")]
                else:
                    labels = [_edge_text(float(v)) for v in ev]
                names = []
                for k in range(nbins):
                    last = k == nbins - 1
                    first = k == 0
                    if right:
                        names.append(f"{'[' if first else '('}{labels[k]}, {labels[k + 1]}]")
                    else:
                        names.append(f"[{labels[k]}, {labels[k + 1]}{']' if last else ')'}")
            result = CategoricalArray(idx, names, ordinal=True)
        elif is_text(third) or _is_time(e) and is_text(third):
            raise MatlabError(f"discretize: unknown option '{text(third)}'.", "MATLAB:discretize:InvalidOption")
        else:
            vals = third
            vdata = vals.data if _is_time(vals) else np.asarray(to_numeric(vals))
            flat = vdata.flatten(order="F")
            if flat.size != nbins:
                raise MatlabError("discretize: give one value for each bin.", "MATLAB:discretize:ValuesInvalidSize")
            out = np.full(idx.shape, np.nan)
            ok = ~np.isnan(idx)
            out[ok] = flat[idx[ok].astype(int) - 1].astype(float)
            result = vals.like(out) if _is_time(vals) else out
    return (result, edges_out) if nargout > 1 else result


# -- stack / unstack ---------------------------------------------------------------------------------------------

@builtin("stack")
def _stack(interp, args, nargout):
    """S = stack(U, vars, ...): several variables of a wide table become one variable of a tall table."""
    t = _need_table(args, "stack")
    rest, opts = _options(list(args[1:]), ("NewDataVariableName", "IndexVariableName", "ConstantVariables"),
                          "stack", 1)
    if not rest:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    stacked = _var_names(t, rest[0])
    if not stacked:
        raise MatlabError("stack: name at least one variable to stack.", "MATLAB:table:stack:NoDataVars")
    const = _var_names(t, opts["ConstantVariables"]) if "ConstantVariables" in opts else \
        [n for n in t.names if n not in stacked]
    data_name = text(opts["NewDataVariableName"]) if "NewDataVariableName" in opts else "_".join(stacked)
    index_name = text(opts["IndexVariableName"]) if "IndexVariableName" in opts else data_name + "_Indicator"
    n, k = t.nrows, len(stacked)
    row_idx = np.repeat(np.arange(n), k)
    from ..ops import concat
    from .categorical import CategoricalArray
    pieces = []
    for i in range(n):
        for name in stacked:
            pieces.append(_rows(t.columns[t.var_index(name)], np.array([i])))
    try:
        data_col = concat(pieces, 0) if pieces else np.zeros((0, 1))
    except MatlabError:
        raise MatlabError("stack: the variables to stack must have types that can be combined in one variable.",
                          "MATLAB:table:stack:IncompatibleTypes") from None
    indicator = CategoricalArray(np.tile(np.arange(1.0, k + 1), n).reshape(-1, 1), list(stacked))
    names = list(const) + [index_name, data_name]
    cols = [_rows(t.columns[t.var_index(c)], row_idx) for c in const] + [indicator, data_col]
    out = t.make(names, cols, None)
    if nargout > 1:
        return out, (row_idx + 1.0).reshape(-1, 1)
    return out


@builtin("unstack")
def _unstack(interp, args, nargout):
    """U = unstack(S, vars, ivar, ...): one variable of a tall table is spread over several variables."""
    t = _need_table(args, "unstack")
    rest, opts = _options(list(args[1:]), ("GroupingVariables", "ConstantVariables", "NewDataVariableNames",
                                           "AggregationFunction", "VariableNamingRule"), "unstack", 2)
    if len(rest) < 2:
        raise MatlabError("unstack needs the data variable and the indicator variable: unstack(S, vars, ivar).",
                          "MATLAB:minrhs")
    data_vars = _var_names(t, rest[0])
    ivar = _var_names(t, rest[1])
    if len(ivar) != 1:
        raise MatlabError("unstack: name exactly one indicator variable.", "MATLAB:table:unstack:MultipleIndVar")
    ivar = ivar[0]
    const = _var_names(t, opts["ConstantVariables"]) if "ConstantVariables" in opts else []
    if "GroupingVariables" in opts:
        group = _var_names(t, opts["GroupingVariables"])
    else:
        group = [n for n in t.names if n not in data_vars and n != ivar and n not in const]
    icol = t.columns[t.var_index(ivar)]
    if getattr(icol, "kind", "") == "categorical":
        present = sorted({int(c) for c in icol.data.flatten() if not np.isnan(c)})
        ikeys = [icol.categories[int(c) - 1] if not np.isnan(c) else _MISSING for c in icol.data[:, 0]]
        levels = [icol.categories[c - 1] for c in present]
    else:
        ikeys = _col_keys(icol)
        levels = sorted({k for k in ikeys if k is not _MISSING}, key=_sort_key)
    if "NewDataVariableNames" in opts:
        new_names = _names_from(opts["NewDataVariableNames"])
        if len(new_names) != len(levels) * len(data_vars):
            raise MatlabError("unstack: give one new variable name for each value of the indicator variable.",
                              "MATLAB:table:unstack:WrongNumNewNames")
    else:
        preserve = "VariableNamingRule" in opts and text(opts["VariableNamingRule"]).lower() == "preserve"
        base = [(_key_text(lv) if preserve else _valid_name(_key_text(lv))) for lv in levels]
        new_names = base if len(data_vars) == 1 else [f"{d}_{b}" for d in data_vars for b in base]
    n = t.nrows
    if group:
        gidx, first = _group_rows([t.columns[t.var_index(g)] for g in group])
    else:
        gidx, first = (np.ones(n), [0]) if n else (np.zeros(0), [])
    ngroups = len(first)
    agg = opts.get("AggregationFunction")
    names = list(group) + list(const)
    pos = np.array(first, dtype=int)
    cols = [_rows(t.columns[t.var_index(g)], pos) for g in group + const]
    k = 0
    for dv in data_vars:
        dcol = t.columns[t.var_index(dv)]
        numeric = isinstance(dcol, np.ndarray) and dcol.dtype.kind in "biuf"
        for lv in levels:
            values = []
            for g in range(ngroups):
                members = [i for i in range(n) if gidx[i] == g + 1 and ikeys[i] == lv]
                if not members:
                    values.append(None)
                    continue
                piece = _rows(dcol, np.array(members))
                if agg is not None:
                    values.append(interp.call_function_value(agg, [piece], 1)[0])
                elif numeric:
                    values.append(np.sum(piece.astype(float), axis=0, keepdims=True))
                elif len(members) == 1:
                    values.append(_row_value(dcol, members[0]))
                else:
                    raise MatlabError("unstack: several rows fall into the same place and the data are not "
                                      "numeric. Give an 'AggregationFunction'.",
                                      "MATLAB:table:unstack:MultipleRows")
            cols.append(_fill_column(values, dcol))
            names.append(new_names[k])
            k += 1
    dup = [x for x in names if names.count(x) > 1]
    if dup:
        raise MatlabError(f"unstack: the new variable name '{dup[0]}' is used twice.",
                          "MATLAB:table:DuplicateVarNames")
    out = t.make(names, cols, None)
    if nargout > 1:
        return out, (pos + 1.0).reshape(-1, 1)
    return out


def _fill_column(values: list, like: Any) -> Any:
    """A column from per-row values where None stands for 'no data' (filled with a missing value)."""
    n = len(values)
    sample = next((v for v in values if v is not None), None)
    if sample is None:
        return np.full((n, 1), np.nan)
    if isinstance(sample, np.ndarray) and not is_char(sample):
        width = sample.reshape(1, -1).shape[1]
        out = np.full((n, width), np.nan)
        for i, v in enumerate(values):
            if v is not None:
                out[i, :] = np.asarray(v, dtype=float).reshape(1, -1)
        return out
    if isinstance(sample, (MString, StringArray)):
        data = np.empty((n, 1), dtype=object)
        for i, v in enumerate(values):
            data[i, 0] = None if v is None else (str(v) if isinstance(v, MString) else v.data.flat[0])
        return StringArray(data)
    if _is_time(sample):
        return sample.like(np.array([[np.nan if v is None else float(v.data.flat[0])] for v in values]))
    data = np.empty((n, 1), dtype=object)
    for i, v in enumerate(values):
        if v is None:
            data[i, 0] = char("") if isinstance(like, Cell) and all(is_char(x) for x in like.data.flat) \
                else np.zeros((0, 0))
        else:
            data[i, 0] = v.data.flat[0] if isinstance(v, Cell) and v.data.size == 1 else v
    return Cell(data)


# -- rowfun / varfun -----------------------------------------------------------------------------------------------

def _output_format(opts: dict, name: str) -> str:
    fmt = text(opts["OutputFormat"]).lower() if "OutputFormat" in opts else "auto"
    if fmt not in ("auto", "table", "timetable", "uniform", "cell"):
        raise MatlabError(f"{name}: 'OutputFormat' must be 'table', 'timetable', 'uniform', 'cell' or 'auto'.",
                          "MATLAB:table:InvalidOutputFormat")
    return fmt


def _call(interp, fn: Any, call_args: list, nout: int, handler: Any, index: int) -> list:
    try:
        res = interp.call_function_value(fn, call_args, nout)
    except MatlabError as e:
        if handler is None:
            raise
        info = Struct.scalar({"message": char(e.message), "identifier": char(e.identifier or ""),
                              "index": np.array([[float(index)]])})
        res = interp.call_function_value(handler, [info, *call_args], nout)
    if len(res) < nout:
        raise MatlabError("The function returned fewer outputs than requested.", "MATLAB:table:TooFewOutputs")
    return res


def _fun_name(fn: Any) -> str:
    name = getattr(fn, "name", None)
    if name and re.fullmatch(r"[A-Za-z]\w*", str(name)):
        return str(name)
    return "Fun"


@builtin("rowfun")
def _rowfun(interp, args, nargout):
    """B = rowfun(func, T, ...): call func once per row (or once per group) with the variables as inputs."""
    if len(args) < 2:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    fn = args[0]
    t = _need_table(args, "rowfun", 1)
    _, opts = _options(list(args[2:]), ("InputVariables", "GroupingVariables", "OutputVariableNames",
                                        "NumOutputs", "SeparateInputs", "ExtractCellContents", "OutputFormat",
                                        "ErrorHandler"), "rowfun")
    groups = _var_names(t, opts["GroupingVariables"]) if "GroupingVariables" in opts else []
    inputs = _var_names(t, opts["InputVariables"]) if "InputVariables" in opts else \
        [n for n in t.names if n not in groups]
    out_names = _names_from(opts["OutputVariableNames"]) if "OutputVariableNames" in opts else None
    nout = int(to_numeric(opts["NumOutputs"]).flat[0]) if "NumOutputs" in opts else (len(out_names) if out_names else 1)
    separate = _flag(opts["SeparateInputs"]) if "SeparateInputs" in opts else True
    extract = _flag(opts["ExtractCellContents"]) if "ExtractCellContents" in opts else False
    fmt = _output_format(opts, "rowfun")
    handler = opts.get("ErrorHandler")
    in_cols = [t.columns[t.var_index(n)] for n in inputs]

    def arguments(rows: list[int]) -> list:
        vals = []
        for c in in_cols:
            if len(rows) == 1 and not groups:
                v = _row_value(c, rows[0])
            else:
                v = _rows(c, np.array(rows, dtype=int))
            if extract and isinstance(v, Cell) and v.data.size == 1:
                v = v.data.flat[0]
            vals.append(v)
        if separate:
            return vals
        from ..ops import concat
        return [concat(vals, 1)] if vals else []

    if groups:
        gidx, first = _group_rows([t.columns[t.var_index(g)] for g in groups])
        members = [[i for i in range(t.nrows) if gidx[i] == g + 1] for g in range(len(first))]
    else:
        members = [[i] for i in range(t.nrows)]
    results: list[list] = [[] for _ in range(nout)]
    for k, rows in enumerate(members):
        res = _call(interp, fn, arguments(rows), nout if nout > 0 else 0, handler, k + 1)
        for j in range(nout):
            results[j].append(res[j])
    if fmt == "uniform":
        outs = []
        for j in range(nout):
            if not all(isinstance(r, np.ndarray) and r.size == 1 for r in results[j]):
                raise MatlabError("With 'OutputFormat' 'uniform' the function must return scalars.",
                                  "MATLAB:table:rowfun:NotAScalarOutput")
            outs.append(_stack_results(results[j]))
        return tuple(outs) if nargout > 1 else outs[0]
    if fmt == "cell":
        data = np.empty((len(members), nout), dtype=object)
        for j in range(nout):
            for i, r in enumerate(results[j]):
                data[i, j] = r
        return Cell(data)
    if out_names is None:
        out_names = [f"Var{j + 1}" for j in range(nout)]
    if len(out_names) != nout:
        raise MatlabError("rowfun: give one output variable name for each output of the function.",
                          "MATLAB:table:rowfun:OutputNamesWrongLength")
    cols = []
    for j in range(nout):
        try:
            cols.append(_stack_results(results[j]))
        except MatlabError:
            cols.append(_column_from_values(results[j]))
    if groups:
        pos = np.array(first, dtype=int)
        names = list(groups) + ["GroupCount"] + out_names
        gcols = [_rows(t.columns[t.var_index(g)], pos) for g in groups]
        counts = np.array([[float(len(m))] for m in members]).reshape(-1, 1)
        return TableObject(names, gcols + [counts] + cols).inherit(t)
    out = t.make(out_names, cols, t.row_names)
    return out


def _varfun(interp, args, nargout):
    """B = varfun(func, T, ...): call func on every variable (or on every variable of every group)."""
    if len(args) < 2:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    fn = args[0]
    t = _need_table(args, "varfun", 1)
    _, opts = _options(list(args[2:]), ("GroupingVariables", "InputVariables", "OutputFormat", "ErrorHandler"),
                       "varfun")
    groups = _var_names(t, opts["GroupingVariables"]) if "GroupingVariables" in opts else []
    if "InputVariables" in opts:
        inputs = _var_names(t, opts["InputVariables"])
    else:
        inputs = [n for n in t.names if n not in groups]
    fmt = _output_format(opts, "varfun")
    handler = opts.get("ErrorHandler")
    fname = _fun_name(fn)
    out_names = [f"{fname}_{n}" for n in inputs]
    if groups:
        gidx, first = _group_rows([t.columns[t.var_index(g)] for g in groups])
        members = [np.array([i for i in range(t.nrows) if gidx[i] == g + 1], dtype=int) for g in range(len(first))]
        cols = []
        for n in inputs:
            col = t.columns[t.var_index(n)]
            res = [_call(interp, fn, [_rows(col, m)], 1, handler, k + 1)[0] for k, m in enumerate(members)]
            try:
                cols.append(_stack_results(res) if res else np.zeros((0, 1)))
            except MatlabError:
                cols.append(_column_from_values(res))
        if fmt == "uniform":
            return np.hstack([np.asarray(to_numeric(c), dtype=float) for c in cols]) if cols else np.zeros((0, 0))
        if fmt == "cell":
            data = np.empty((len(members), len(cols)), dtype=object)
            for j, c in enumerate(cols):
                for i in range(len(members)):
                    data[i, j] = _row_value(c, i)
            return Cell(data)
        pos = np.array(first, dtype=int)
        counts = np.array([[float(m.size)] for m in members]).reshape(-1, 1)
        return TableObject(list(groups) + ["GroupCount"] + out_names,
                           [_rows(t.columns[t.var_index(g)], pos) for g in groups] + [counts] + cols).inherit(t)
    results = [_call(interp, fn, [t.columns[t.var_index(n)]], 1, handler, k + 1)[0] for k, n in enumerate(inputs)]
    if fmt == "uniform":
        if not all(isinstance(r, np.ndarray) and r.size == 1 for r in results):
            raise MatlabError("With 'OutputFormat' 'uniform' the function must return scalars.",
                              "MATLAB:table:varfun:NotAScalarOutput")
        dtype = results[0].dtype if results and len({r.dtype for r in results}) == 1 else float
        return np.array([[r.flat[0] for r in results]], dtype=dtype).reshape(1, -1) if results else np.zeros((1, 0))
    if fmt == "cell":
        return Cell.from_list(results)
    cols = [_as_column(r) for r in results]
    heights = {_height(c) for c in cols}
    if len(heights) > 1:
        raise MatlabError("varfun: the function must return values with the same number of rows for every "
                          "variable. Use 'OutputFormat', 'cell' for other results.",
                          "MATLAB:table:varfun:UnequalRows")
    keep_rows = t.row_names if heights == {t.nrows} else None
    return t.make(out_names, cols, keep_rows)


TableObject._METHODS = TableObject._METHODS | {"varfun"}
_TABLE_FUNCS["varfun"] = _varfun
builtin("varfun")(_varfun)


# -- group functions ------------------------------------------------------------------------------------------------

_TIME_BINS = ("second", "minute", "hour", "day", "week", "month", "quarter", "year", "decade", "century",
              "secondofminute", "minuteofhour", "hourofday", "dayname", "dayofweek", "dayofmonth", "dayofyear",
              "weekofmonth", "weekofyear", "monthname", "monthofyear", "quarterofyear")


def _binned_column(interp, col: Any, spec: Any, right: bool = False) -> Any:
    """A grouping variable put into bins: numeric edges, a number of bins or a time unit."""
    if is_text(spec):
        word = text(spec).lower()
        if word == "none":
            return col
        if getattr(col, "kind", "") == "datetime" and word in _TIME_BINS:
            from .categorical import CategoricalArray
            if word in ("second", "minute", "hour", "day", "week", "month", "quarter", "year"):
                start = BUILTINS["dateshift"](interp, [col, char("start"), char(word)], 1)
                fmts = {"year": "yyyy", "quarter": "QQQ yyyy", "month": "MMM-yyyy", "day": "dd-MMM-yyyy",
                        "week": "dd-MMM-yyyy", "hour": "dd-MMM-yyyy HH:mm", "minute": "dd-MMM-yyyy HH:mm",
                        "second": "dd-MMM-yyyy HH:mm:ss"}
                keys = start.data[:, 0]
                order = sorted({float(k) for k in keys if k == k})
                names = [str(start.like(np.array([[k]])).element_texts(fmts[word]).flat[0]) for k in order]
                codes = np.array([order.index(float(k)) + 1.0 if k == k else np.nan for k in keys]).reshape(-1, 1)
                return CategoricalArray(codes, names, ordinal=True)
            pick = {"hourofday": ("hour", []), "dayofmonth": ("day", []), "dayofyear": ("day", ["dayofyear"]),
                    "dayofweek": ("day", ["dayofweek"]), "dayname": ("day", ["name"]),
                    "monthname": ("month", ["name"]), "monthofyear": ("month", []),
                    "quarterofyear": ("quarter", []), "weekofyear": ("week", []),
                    "weekofmonth": ("week", ["weekofmonth"]), "minuteofhour": ("minute", []),
                    "secondofminute": ("second", []), "decade": ("year", []), "century": ("year", [])}[word]
            vals = BUILTINS[pick[0]](interp, [col] + [char(a) for a in pick[1]], 1)
            if word == "decade":
                vals = np.floor(vals / 10) * 10
            elif word == "century":
                vals = np.floor(vals / 100) * 100
            if isinstance(vals, Cell):
                order_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] \
                    if word == "dayname" else ["January", "February", "March", "April", "May", "June", "July",
                                               "August", "September", "October", "November", "December"]
                texts = [to_str(x) for x in vals.data.flatten(order="F")]
                present = [nm for nm in order_names if nm in texts]
                codes = np.array([present.index(x) + 1.0 if x in present else np.nan for x in texts]).reshape(-1, 1)
                return CategoricalArray(codes, present, ordinal=True)
            return np.asarray(vals, dtype=float).reshape(-1, 1)
        raise MatlabError(f"Unknown group bins '{text(spec)}'.", "MATLAB:groupsummary:InvalidGroupBins")
    extra = [char("IncludedEdge"), char("right")] if right else []
    return _discretize(interp, [col, spec, char("categorical"), *extra], 1)


def _bin_specs(spec: Any, n: int, name: str) -> list:
    if isinstance(spec, Cell):
        items = spec.values_f()
        if len(items) != n:
            raise MatlabError(f"{name}: give one set of bins for each grouping variable.",
                              "MATLAB:groupsummary:GroupBinsInvalidSize")
        return items
    return [spec] * n


def _is_binspec(v: Any) -> bool:
    if isinstance(v, Cell):
        return all(_is_binspec(x) for x in v.values_f()) and v.data.size > 0
    if is_text(v):
        return text(v).lower() in _TIME_BINS or text(v).lower() == "none"
    return (isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and not is_char(v)) or \
        getattr(v, "kind", "") in ("duration", "datetime", "calendarDuration")


def _grouping(interp, t: TableObject, rest: list, name: str, opts: dict) -> tuple[list[str], list, list]:
    """(names of the group columns in the result, the group columns, the remaining arguments)."""
    group_names = _var_names(t, rest[0])
    cols = [t.columns[t.var_index(g)] for g in group_names]
    rest = rest[1:]
    out_names = list(group_names)
    right = "IncludedEdge" in opts and text(opts["IncludedEdge"]).lower() == "right"
    if rest and _is_binspec(rest[0]) and not (is_text(rest[0]) and text(rest[0]).lower() in _SUMMARY_WORDS):
        specs = _bin_specs(rest[0], len(group_names), name)
        rest = rest[1:]
        new_cols, out_names = [], []
        for g, c, s in zip(group_names, cols, specs):
            b = _binned_column(interp, c, s, right)
            new_cols.append(b)
            if b is c:
                out_names.append(g)
            elif is_text(s):
                out_names.append(f"{text(s).lower()}_{g}")
            else:
                out_names.append(f"disc_{g}")
        cols = new_cols
    return out_names, cols, rest


_SUMMARY_WORDS = {"mean", "sum", "min", "max", "range", "median", "mode", "var", "std", "nummissing", "nnz",
                  "numunique", "all", "numel"}


def _group_members(cols: list, n: int, include_missing: bool) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Row numbers of each group, the first row of each group, and the group number of every row."""
    if not cols:
        return ([np.arange(n)] if n else []), np.array([0] if n else [], dtype=int), np.ones(n)
    gidx, first = _group_rows(cols)
    members = [np.nonzero(gidx == g + 1)[0] for g in range(len(first))]
    firsts = list(first)
    if include_missing:
        # rows with a missing key form extra groups (one per combination of the other keys)
        keys = [_col_keys(c) for c in cols]
        extra: dict[tuple, list[int]] = {}
        for i in range(n):
            if np.isnan(gidx[i]):
                key = tuple(("\0missing",) if k[i] is _MISSING else (0, k[i]) for k in keys)
                extra.setdefault(key, []).append(i)
        for key in sorted(extra, key=lambda kk: tuple((1, 0) if p == ("\0missing",) else (0,) + tuple(
                _sort_key(p[1])) for p in kk)):
            rows = extra[key]
            gidx[rows] = len(members) + 1
            members.append(np.array(rows, dtype=int))
            firsts.append(rows[0])
    return members, np.array(firsts, dtype=int), gidx


@builtin("groupcounts")
def _groupcounts(interp, args, nargout):
    """G = groupcounts(T, groupvars[, groupbins]) or [B, BG] = groupcounts(A)."""
    if not args:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    rest, opts = _options(list(args), ("IncludeMissingGroups", "IncludeEmptyGroups", "IncludePercentGroups",
                                       "IncludedEdge"), "groupcounts", 1)
    include_missing = _flag(opts["IncludeMissingGroups"]) if "IncludeMissingGroups" in opts else True
    if not isinstance(rest[0], TableObject):
        vals = rest[0]
        cols = [_as_column(x) for x in vals.values_f()] if isinstance(vals, Cell) and not all(
            is_char(x) for x in vals.data.flat) else [_as_column(vals)]
        if len(rest) > 1:
            specs = _bin_specs(rest[1], len(cols), "groupcounts")
            cols = [_binned_column(interp, c, s) for c, s in zip(cols, specs)]
        n = _height(cols[0])
        members, firsts, _ = _group_members(cols, n, include_missing)
        counts = np.array([[float(m.size)] for m in members]).reshape(-1, 1)
        groups = [_rows(c, firsts) for c in cols]
        bg = groups[0] if len(groups) == 1 else Cell.from_list(groups)
        pct = 100.0 * counts / max(float(n), 1.0)
        return (counts, bg, pct)[:max(nargout, 1)] if nargout > 1 else counts
    t = rest[0]
    if len(rest) < 2:
        raise MatlabError("groupcounts needs the grouping variables: groupcounts(T, groupvars).", "MATLAB:minrhs")
    out_names, cols, _ = _grouping(interp, t, rest[1:], "groupcounts", opts)
    members, firsts, _ = _group_members(cols, t.nrows, include_missing)
    counts = np.array([[float(m.size)] for m in members]).reshape(-1, 1)
    total = float(sum(m.size for m in members)) or 1.0
    names = out_names + ["GroupCount", "Percent"]
    return TableObject(names, [_rows(c, firsts) for c in cols] + [counts, 100.0 * counts / total]).inherit(t)


def _apply_summary(interp, method: Any, piece: Any) -> Any:
    from .table import _apply_method
    if isinstance(method, str) and method == "numunique":
        keys = [k for k in _col_keys(piece) if k is not _MISSING]
        return np.array([[float(len(set(keys)))]])
    if isinstance(method, str) and method == "nummissing":
        m = _missing_mask(piece)
        return np.sum(m.reshape(m.shape[0], -1), axis=0, keepdims=True).astype(float)
    if isinstance(method, str) and _is_time(piece) and method in ("mean", "min", "max", "median", "mode", "range",
                                                                    "sum", "std", "var"):
        res = interp.call_named(method, [piece], 1, interp.frame)
        return res[0]
    r = _apply_method(interp, method, piece)
    return np.array([[float(r)]]) if isinstance(r, (int, float)) else r


@builtin("groupsummary")
def _groupsummary(interp, args, nargout):
    """G = groupsummary(T, groupvars[, groupbins][, method[, datavars]]) or B = groupsummary(A, groupvars, method)."""
    if len(args) < 2:
        raise MatlabError("groupsummary expects (T, groupvars, ...).", "MATLAB:minrhs")
    rest, opts = _options(list(args), ("IncludeMissingGroups", "IncludeEmptyGroups", "IncludedEdge"),
                          "groupsummary", 2)
    include_missing = _flag(opts["IncludeMissingGroups"]) if "IncludeMissingGroups" in opts else True
    if not isinstance(rest[0], TableObject):
        x = rest[0]
        gv = rest[1]
        cols = [_as_column(c) for c in gv.values_f()] if isinstance(gv, Cell) and not all(
            is_char(c) for c in gv.data.flat) else [_as_column(gv)]
        more = rest[2:]
        if more and _is_binspec(more[0]) and not (is_text(more[0]) and text(more[0]).lower() in _SUMMARY_WORDS):
            specs = _bin_specs(more[0], len(cols), "groupsummary")
            cols = [_binned_column(interp, c, s) for c, s in zip(cols, specs)]
            more = more[1:]
        methods = _method_list(more[0]) if more else []
        xc = _as_column(x)
        members, firsts, _ = _group_members(cols, _height(xc), include_missing)
        blocks = []
        for method in methods:
            res = [_apply_summary(interp, method, _rows(xc, m)) for m in members]
            blocks.append(_stack_results(res) if res else np.zeros((0, 1)))
        b = np.hstack([np.asarray(to_numeric(c), dtype=float) for c in blocks]) if blocks else \
            np.zeros((len(members), 0))
        groups = [_rows(c, firsts) for c in cols]
        bg = groups[0] if len(groups) == 1 else Cell.from_list(groups)
        counts = np.array([[float(m.size)] for m in members]).reshape(-1, 1)
        return (b, bg, counts)[:max(nargout, 1)] if nargout > 1 else b
    t = rest[0]
    out_names, cols, more = _grouping(interp, t, rest[1:], "groupsummary", opts)
    group_vars = _var_names(t, rest[1])
    methods = _method_list(more[0]) if more else []
    if len(more) > 1:
        data_names = _var_names(t, more[1])
    else:
        data_names = [n for n, c in zip(t.names, t.columns) if n not in group_vars and
                      (_is_numeric_var(c) or getattr(c, "kind", "") == "duration")] if methods else []
    members, firsts, _ = _group_members(cols, t.nrows, include_missing)
    names = list(out_names) + ["GroupCount"]
    out_cols = [_rows(c, firsts) for c in cols] + [np.array([[float(m.size)] for m in members]).reshape(-1, 1)]
    for dn in data_names:
        col = t.columns[t.var_index(dn)]
        for k, method in enumerate(methods):
            res = [_apply_summary(interp, method, _rows(col, m)) for m in members]
            label = f"fun{k + 1}" if isinstance(method, FunctionHandle) else method
            names.append(f"{label}_{dn}")
            try:
                out_cols.append(_stack_results(res) if res else np.zeros((0, 1)))
            except MatlabError:
                out_cols.append(_column_from_values(res))
    return TableObject(names, out_cols).inherit(t)


@builtin("groupfilter")
def _groupfilter(interp, args, nargout):
    """G = groupfilter(T, groupvars, method[, datavars]): the rows of the groups for which method is true."""
    if len(args) < 3:
        raise MatlabError("groupfilter needs (T, groupvars, method).", "MATLAB:minrhs")
    rest, opts = _options(list(args), ("IncludedEdge",), "groupfilter", 3)
    if not isinstance(rest[0], TableObject):
        x = _as_column(rest[0])
        gv = rest[1]
        cols = [_as_column(c) for c in gv.values_f()] if isinstance(gv, Cell) and not all(
            is_char(c) for c in gv.data.flat) else [_as_column(gv)]
        members, _, _ = _group_members(cols, _height(x), True)
        keep = np.zeros(_height(x), dtype=bool)
        for m in members:
            res = interp.call_function_value(rest[2], [_rows(x, m)], 1)[0]
            flags = np.asarray(to_numeric(res)).astype(bool)
            keep[m] = flags.flatten() if flags.size == m.size else bool(flags.all())
        idx = np.nonzero(keep)[0]
        out = _rows(x, idx)
        groups = [_rows(c, idx) for c in cols]
        return (out, groups[0] if len(groups) == 1 else Cell.from_list(groups)) if nargout > 1 else out
    t = rest[0]
    out_names, cols, more = _grouping(interp, t, rest[1:], "groupfilter", opts)
    group_vars = _var_names(t, rest[1])
    if not more:
        raise MatlabError("groupfilter needs a filtering function.", "MATLAB:minrhs")
    method = more[0]
    if not isinstance(method, FunctionHandle):
        raise MatlabError("groupfilter: the method must be a function handle that returns true or false.",
                          "MATLAB:groupfilter:InvalidMethod")
    data_names = _var_names(t, more[1]) if len(more) > 1 else [n for n in t.names if n not in group_vars]
    members, _, _ = _group_members(cols, t.nrows, True)
    keep = np.ones(t.nrows, dtype=bool)
    for dn in data_names:
        col = t.columns[t.var_index(dn)]
        for m in members:
            res = interp.call_function_value(method, [_rows(col, m)], 1)[0]
            flags = np.asarray(to_numeric(res)).astype(bool)
            if flags.size == m.size and m.size != 1:
                keep[m] &= flags.flatten(order="F")
            elif flags.size == 1:
                keep[m] &= bool(flags.flat[0])
            else:
                raise MatlabError("groupfilter: the function must return one logical value per group or one "
                                  "per row of the group.", "MATLAB:groupfilter:InvalidMethodOutput")
    idx = np.nonzero(keep)[0]
    out = t.index_get(interp, [idx.reshape(-1, 1) + 1.0, ix.COLON])
    binned = [(nm, c) for nm, c, g in zip(out_names, cols, group_vars) if nm != g]
    for nm, c in binned:
        out = out.set_field(interp, nm, _rows(c, idx))
    return out


def _transform(interp, method: Any, piece: Any) -> Any:
    if isinstance(method, FunctionHandle):
        return interp.call_function_value(method, [piece], 1)[0]
    a = np.asarray(to_numeric(piece), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "zscore":
            sd = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])
            return (a - np.nanmean(a, axis=0)) / np.where(sd == 0, 1.0, sd) if a.shape[0] else a
        if method == "norm":
            nrm = np.sqrt(np.nansum(a ** 2, axis=0))
            return a / np.where(nrm == 0, 1.0, nrm)
        if method == "meancenter":
            return a - np.nanmean(a, axis=0)
        if method == "rescale":
            lo, hi = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
            span = np.where(hi == lo, 1.0, hi - lo)
            return np.where(hi == lo, 0.0, (a - lo) / span)
        if method == "meanfill":
            return np.where(np.isnan(a), np.nanmean(a, axis=0), a)
        if method == "linearfill":
            out = a.copy()
            for j in range(a.shape[1]):
                col = out[:, j]
                ok = ~np.isnan(col)
                if ok.sum() >= 2:
                    xs = np.arange(col.size)
                    slope_l = (col[ok][1] - col[ok][0]) / (xs[ok][1] - xs[ok][0])
                    slope_r = (col[ok][-1] - col[ok][-2]) / (xs[ok][-1] - xs[ok][-2])
                    filled = np.interp(xs, xs[ok], col[ok])
                    left = xs < xs[ok][0]
                    right = xs > xs[ok][-1]
                    filled[left] = col[ok][0] + slope_l * (xs[left] - xs[ok][0])
                    filled[right] = col[ok][-1] + slope_r * (xs[right] - xs[ok][-1])
                    out[:, j] = filled
            return out
    raise MatlabError(f"Unknown grouptransform method '{method}'. Use 'zscore', 'norm', 'meancenter', 'rescale', "
                      "'meanfill', 'linearfill' or a function handle.", "MATLAB:grouptransform:InvalidMethod")


@builtin("grouptransform")
def _grouptransform(interp, args, nargout):
    """G = grouptransform(T, groupvars, method[, datavars], 'ReplaceValues', tf)."""
    if len(args) < 3:
        raise MatlabError("grouptransform needs (T, groupvars, method).", "MATLAB:minrhs")
    rest, opts = _options(list(args), ("ReplaceValues", "IncludedEdge"), "grouptransform", 3)
    replace = _flag(opts["ReplaceValues"]) if "ReplaceValues" in opts else True
    if not isinstance(rest[0], TableObject):
        x = _as_column(rest[0])
        gv = rest[1]
        cols = [_as_column(c) for c in gv.values_f()] if isinstance(gv, Cell) and not all(
            is_char(c) for c in gv.data.flat) else [_as_column(gv)]
        method = rest[2] if isinstance(rest[2], FunctionHandle) else text(rest[2]).lower()
        members, _, _ = _group_members(cols, _height(x), True)
        out = np.asarray(to_numeric(x), dtype=float).copy()
        for m in members:
            res = np.asarray(to_numeric(_transform(interp, method, _rows(x, m))), dtype=float)
            out[m, ...] = res if res.shape[0] == m.size else np.broadcast_to(res, out[m, ...].shape)
        shaped = out.reshape(np.shape(rest[0])) if isinstance(rest[0], np.ndarray) else out
        return (shaped, rest[1]) if nargout > 1 else shaped
    t = rest[0]
    out_names, cols, more = _grouping(interp, t, rest[1:], "grouptransform", opts)
    group_vars = _var_names(t, rest[1])
    if not more:
        raise MatlabError("grouptransform needs a method.", "MATLAB:minrhs")
    method = more[0] if isinstance(more[0], FunctionHandle) else text(more[0]).lower()
    if len(more) > 1:
        data_names = _var_names(t, more[1])
    else:
        data_names = [n for n, c in zip(t.names, t.columns) if n not in group_vars and _is_numeric_var(c)]
    members, _, _ = _group_members(cols, t.nrows, True)
    out = t.copy()
    label = "fun" if isinstance(method, FunctionHandle) else method
    for dn in data_names:
        col = t.columns[t.var_index(dn)]
        new = np.asarray(to_numeric(col), dtype=float).copy() if isinstance(col, np.ndarray) else None
        if new is None:
            raise MatlabError(f"grouptransform: the variable '{dn}' must be numeric.",
                              "MATLAB:grouptransform:NonNumericData")
        for m in members:
            res = np.asarray(to_numeric(_transform(interp, method, _rows(col, m))), dtype=float)
            if res.shape[0] == m.size:
                new[m, ...] = res.reshape(new[m, ...].shape)
            elif res.size == 1 or res.shape[0] == 1:
                new[m, ...] = np.broadcast_to(res.reshape(1, -1), new[m, ...].shape)
            else:
                raise MatlabError("grouptransform: the function must return one row per row of the group, "
                                  "or a single row.", "MATLAB:grouptransform:InvalidMethodOutput")
        out = out.set_field(interp, dn if replace else f"{label}_{dn}", new)
    for nm, c, g in zip(out_names, cols, group_vars):
        if nm != g:
            out = out.set_field(interp, nm, c)
    return out


# -- pivot -------------------------------------------------------------------------------------------------------

_PIVOT_METHODS = ("count", "sum", "mean", "median", "min", "max", "mode", "std", "var", "range", "numunique",
                  "nummissing", "nnz", "percentage", "none")


@builtin("pivot")
def _pivot(interp, args, nargout):
    """P = pivot(T, Columns=colvars, Rows=rowvars, DataVariable=dv, Method=method, ...)."""
    t = _need_table(args, "pivot")
    _, opts = _options(list(args[1:]), ("Columns", "Rows", "DataVariable", "Method", "IncludeTotals",
                                        "IncludeMissingGroups", "IncludeEmptyGroups", "RowLabelPlacement",
                                        "OutputFormat", "IncludedEdge", "ColumnsBinMethod", "RowsBinMethod"),
                       "pivot")
    col_vars = _var_names(t, opts["Columns"]) if "Columns" in opts else []
    row_vars = _var_names(t, opts["Rows"]) if "Rows" in opts else []
    if not col_vars and not row_vars:
        raise MatlabError("pivot needs 'Columns', 'Rows' or both.", "MATLAB:pivot:NoGroupingVars")
    data_var = _var_names(t, opts["DataVariable"])[0] if "DataVariable" in opts else None
    method: Any = opts.get("Method")
    if method is None:
        method = "count" if data_var is None else "sum"
    elif not isinstance(method, FunctionHandle):
        method = text(method).lower()
        if method not in _PIVOT_METHODS:
            raise MatlabError(f"pivot: unknown method '{method}'.", "MATLAB:pivot:InvalidMethod")
    totals = _flag(opts["IncludeTotals"]) if "IncludeTotals" in opts else False
    include_missing = _flag(opts["IncludeMissingGroups"]) if "IncludeMissingGroups" in opts else True
    n = t.nrows

    def binned(names: list[str], key: str) -> list:
        cols = [t.columns[t.var_index(v)] for v in names]
        if key in opts:
            specs = _bin_specs(opts[key], len(cols), "pivot")
            cols = [_binned_column(interp, c, s) for c, s in zip(cols, specs)]
        return cols

    rcols, ccols = binned(row_vars, "RowsBinMethod"), binned(col_vars, "ColumnsBinMethod")
    rmembers, rfirst, rid = _group_members(rcols, n, include_missing)
    cmembers, cfirst, cid = _group_members(ccols, n, include_missing)
    data_col = t.columns[t.var_index(data_var)] if data_var else None

    def value(rows: np.ndarray) -> Any:
        if isinstance(method, FunctionHandle):
            if data_col is None:
                raise MatlabError("pivot: a function method needs a 'DataVariable'.", "MATLAB:pivot:NoDataVariable")
            return interp.call_function_value(method, [_rows(data_col, rows)], 1)[0]
        if method == "count":
            if data_col is None:
                return np.array([[float(rows.size)]])
            return np.array([[float(np.sum(~_missing_mask(_rows(data_col, rows)).reshape(rows.size, -1).any(axis=1)))]])
        if method == "percentage":
            return np.array([[100.0 * rows.size / max(n, 1)]])
        if data_col is None:
            raise MatlabError(f"pivot: the method '{method}' needs a 'DataVariable'.", "MATLAB:pivot:NoDataVariable")
        if rows.size == 0:
            return np.array([[0.0 if method in ("sum", "nnz", "numunique", "nummissing") else np.nan]])
        if method == "none":
            if rows.size != 1:
                raise MatlabError("pivot: with the method 'none' every group must have exactly one row.",
                                  "MATLAB:pivot:NoneMethodMultipleRows")
            return _row_value(data_col, int(rows[0]))
        return _apply_summary(interp, method, _rows(data_col, rows))

    def col_label(k: int) -> str:
        parts = []
        for c in ccols:
            key = _col_keys(_rows(c, np.array([cfirst[k]])))[0]
            parts.append("<missing>" if key is _MISSING else _label_of(c, int(cfirst[k]), key))
        return "_".join(parts)

    names = list(row_vars)
    cols = [_rows(c, rfirst) for c in rcols]
    row_sets = rmembers if rcols else [np.arange(n)]
    col_sets = cmembers if ccols else [np.arange(n)]
    labels = [col_label(k) for k in range(len(cmembers))] if ccols else [str(method) if isinstance(method, str)
                                                                         else "fun"]
    if ccols and len(labels) == 0:
        labels = []
    grid = [[value(np.intersect1d(r, c)) for c in col_sets] for r in row_sets]
    for j, label in enumerate(labels):
        names.append(label)
        cols.append(_column_from_values([grid[i][j] for i in range(len(row_sets))]))
    if totals:
        names.append("Overall_Total")
        cols.append(_column_from_values([value(r) for r in row_sets]))
    dup = [x for x in names if names.count(x) > 1]
    if dup:
        raise MatlabError(f"pivot: the result would have two variables called '{dup[0]}'.",
                          "MATLAB:table:DuplicateVarNames")
    out = TableObject(names, cols)
    if totals and rcols:
        total_cells = []
        for j, _label in enumerate(labels):
            total_cells.append(value(col_sets[j]))
        total_cells.append(value(np.arange(n)))
        extra_cols = []
        for c in rcols:
            first_row = _rows(c, rfirst[:1]) if len(rfirst) else c
            extra_cols.append(_total_label(first_row))
        tail = TableObject(names, extra_cols + [_column_from_values([v]) for v in total_cells])
        try:
            out = TableObject.concat([out, tail], 0)
        except MatlabError:
            pass
    return out


def _label_of(col: Any, row: int, key: Any) -> str:
    if hasattr(col, "element_texts"):
        return str(np.asarray(col.element_texts(), dtype=object)[row].flat[0])
    return _key_text(key)


def _total_label(like: Any) -> Any:
    """A one-row value for the 'Overall_Total' line in a row-label column."""
    if isinstance(like, Cell):
        d = np.empty((1, 1), dtype=object)
        d[0, 0] = char("Overall_Total")
        return Cell(d)
    if isinstance(like, StringArray):
        d = np.empty((1, 1), dtype=object)
        d[0, 0] = "Overall_Total"
        return StringArray(d)
    if getattr(like, "kind", "") == "categorical":
        from .categorical import CategoricalArray
        return CategoricalArray(np.array([[len(like.categories) + 1.0]]), list(like.categories) + ["Overall_Total"],
                                like.ordinal)
    if _is_time(like):
        return like.like(np.array([[np.nan]]))
    return np.array([[np.nan]])


# -- variables: merge, split, convert, add, rows2vars, inner2outer -----------------------------------------------------

@builtin("mergevars")
def _mergevars(interp, args, nargout):
    """T2 = mergevars(T, vars, 'NewVariableName', name, 'MergeAsTable', tf)."""
    t = _need_table(args, "mergevars")
    rest, opts = _options(list(args[1:]), ("NewVariableName", "MergeAsTable"), "mergevars", 1)
    if not rest:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    vars_ = _var_names(t, rest[0])
    if not vars_:
        return t
    as_table = _flag(opts["MergeAsTable"]) if "MergeAsTable" in opts else False
    where = min(t.var_index(v) for v in vars_)
    name = text(opts["NewVariableName"]) if "NewVariableName" in opts else f"Var{where + 1}"
    parts = [t.columns[t.var_index(v)] for v in vars_]
    if as_table:
        merged: Any = TableObject(list(vars_), parts)
    else:
        from ..ops import concat
        try:
            merged = concat(parts, 1)
        except MatlabError:
            raise MatlabError("mergevars: the variables cannot be combined in one array; use 'MergeAsTable', "
                              "true.", "MATLAB:table:mergevars:IncompatibleTypes") from None
    names, cols = [], []
    for k, (n, c) in enumerate(zip(t.names, t.columns)):
        if k == where:
            names.append(name)
            cols.append(merged)
        if n not in vars_:
            names.append(n)
            cols.append(c)
    if len(set(names)) != len(names):
        raise MatlabError(f"mergevars: the name '{name}' is already a variable of the table.",
                          "MATLAB:table:DuplicateVarNames")
    return t.make(names, cols, t.row_names)


def _split_one(name: str, col: Any, new_names: list[str] | None) -> tuple[list[str], list]:
    if isinstance(col, TableObject):
        return (new_names or list(col.names)), list(col.columns)
    if isinstance(col, (Cell, StringArray)):
        data = col.data
        if data.ndim != 2 or data.shape[1] <= 1:
            return [name], [col]
        parts = [type(col)(data[:, j:j + 1].copy()) for j in range(data.shape[1])]
    elif _is_time(col):
        if col.data.ndim != 2 or col.data.shape[1] <= 1:
            return [name], [col]
        parts = [col.like(col.data[:, j:j + 1].copy()) for j in range(col.data.shape[1])]
    elif isinstance(col, np.ndarray) and not is_char(col):
        if col.ndim != 2 or col.shape[1] <= 1:
            return [name], [col]
        parts = [col[:, j:j + 1].copy() for j in range(col.shape[1])]
    else:
        return [name], [col]
    names = new_names or [f"{name}_{j + 1}" for j in range(len(parts))]
    return names, parts


@builtin("splitvars")
def _splitvars(interp, args, nargout):
    """T2 = splitvars(T[, vars], 'NewVariableNames', names)."""
    t = _need_table(args, "splitvars")
    rest, opts = _options(list(args[1:]), ("NewVariableNames",), "splitvars")
    targets = _var_names(t, rest[0]) if rest else list(t.names)
    new = opts.get("NewVariableNames")
    per_var: dict[str, list[str]] = {}
    if new is not None:
        if isinstance(new, Cell) and new.data.size and all(isinstance(x, (Cell, StringArray)) for x in new.data.flat):
            groups = [_names_from(x) for x in new.values_f()]
            if len(groups) != len(targets):
                raise MatlabError("splitvars: give one list of new names for each variable that is split.",
                                  "MATLAB:table:splitvars:WrongNumNewNames")
            per_var = dict(zip(targets, groups))
        elif len(targets) == 1:
            per_var = {targets[0]: _names_from(new)}
        else:
            raise MatlabError("splitvars: with several variables, 'NewVariableNames' must be a cell array with "
                              "one list of names per variable.", "MATLAB:table:splitvars:WrongNumNewNames")
    names, cols = [], []
    for n, c in zip(t.names, t.columns):
        if n in targets:
            nn, cc = _split_one(n, c, per_var.get(n))
            if len(nn) != len(cc):
                raise MatlabError(f"splitvars: '{n}' splits into {len(cc)} variables but {len(nn)} names were "
                                  "given.", "MATLAB:table:splitvars:WrongNumNewNames")
            names += nn
            cols += cc
        else:
            names.append(n)
            cols.append(c)
    dup = [x for x in names if names.count(x) > 1]
    if dup:
        raise MatlabError(f"splitvars: the result would have two variables called '{dup[0]}'.",
                          "MATLAB:table:DuplicateVarNames")
    return t.make(names, cols, t.row_names)


@builtin("convertvars")
def _convertvars(interp, args, nargout):
    """T2 = convertvars(T, vars, dataType): convert variables with a type name or a function handle."""
    t = _need_table(args, "convertvars")
    nargs(args, 3, 3, "convertvars")
    targets = _var_names(t, args[1])
    conv = args[2]
    out = t.copy()
    for n in targets:
        col = t.columns[t.var_index(n)]
        if isinstance(conv, FunctionHandle):
            res = interp.call_function_value(conv, [col], 1)
        else:
            res = interp.call_named(text(conv, "data type"), [col], 1, interp.frame)
        if not res:
            raise MatlabError("convertvars: the conversion function must return a value.")
        new = _as_column(res[0])
        if _height(new) != t.nrows:
            raise MatlabError(f"convertvars: converting '{n}' changed its number of rows.",
                              "MATLAB:table:convertvars:WrongHeight")
        out.columns[out.names.index(n)] = new
    return out


def _addvars(interp, args, nargout):
    """T2 = addvars(T, var1, ..., 'Before' | 'After', location, 'NewVariableNames', names)."""
    t = _need_table(args, "addvars")
    rest, opts = _options(list(args[1:]), ("NewVariableNames", "Before", "After"), "addvars", 1)
    if "Before" in opts and "After" in opts:
        raise MatlabError("addvars: give 'Before' or 'After', not both.", "MATLAB:table:addmovevars:BeforeAndAfter")
    arg_names = (interp.state.get("call_arg_names") or [])[1:1 + len(rest)]
    if "NewVariableNames" in opts:
        names = _names_from(opts["NewVariableNames"])
        if len(names) != len(rest):
            raise MatlabError("addvars: give one new variable name for each added variable.",
                              "MATLAB:table:addvars:WrongNumNewNames")
    else:
        names = []
        for k in range(len(rest)):
            given = arg_names[k] if k < len(arg_names) else None
            names.append(given if given and given not in t.names and given not in names
                         else f"Var{len(t.names) + k + 1}")
    cols = []
    for n, v in zip(names, rest):
        if n in t.names:
            raise MatlabError(f"addvars: the table already has a variable called '{n}'.",
                              "MATLAB:table:DuplicateVarNames")
        c = _as_column(v)
        if t.names and _height(c) != t.nrows:
            raise MatlabError("addvars: every new variable must have one row for each row of the table.",
                              "MATLAB:table:addvars:WrongHeight")
        cols.append(c)
    where = len(t.names)
    if "Before" in opts:
        where = t.var_positions(opts["Before"])[0]
    elif "After" in opts:
        where = t.var_positions(opts["After"])[0] + 1
    all_names = t.names[:where] + names + t.names[where:]
    all_cols = t.columns[:where] + cols + t.columns[where:]
    return t.make(all_names, all_cols, t.row_names)


def _rows2vars(interp, args, nargout):
    """T2 = rows2vars(T, 'VariableNamesSource', v, 'DataVariables', vars, 'VariableNamingRule', rule)."""
    t = _need_table(args, "rows2vars")
    _, opts = _options(list(args[1:]), ("VariableNamesSource", "DataVariables", "VariableNamingRule"), "rows2vars")
    source = _var_names(t, opts["VariableNamesSource"])[0] if "VariableNamesSource" in opts else None
    data_vars = _var_names(t, opts["DataVariables"]) if "DataVariables" in opts else \
        [n for n in t.names if n != source]
    if source is not None and source in data_vars:
        data_vars = [n for n in data_vars if n != source]
    preserve = "VariableNamingRule" in opts and text(opts["VariableNamingRule"]).lower() == "preserve"
    if source is not None:
        keys = _col_keys(t.columns[t.var_index(source)])
        new = [_key_text(k) if k is not _MISSING else "" for k in keys]
        new = [n if preserve else _valid_name(n) for n in new]
    elif t.row_names:
        new = list(t.row_names) if preserve else [_valid_name(r) for r in t.row_names]
    else:
        new = [f"Var{k + 1}" for k in range(t.nrows)]
    if len(set(new)) != len(new):
        raise MatlabError("rows2vars: the new variable names must be unique.", "MATLAB:table:DuplicateVarNames")
    cols = [t.columns[t.var_index(n)] for n in data_vars]
    names_col = Cell.from_list([char(n) for n in data_vars], column=True)
    out_cols: list = [names_col]
    uniform = all(isinstance(c, np.ndarray) and not is_char(c) and c.ndim == 2 and c.shape[1] == 1 for c in cols)
    for i in range(t.nrows):
        if uniform and cols:
            out_cols.append(np.array([[float(np.real(c[i, 0]))] for c in cols]) if all(
                c.dtype.kind in "biuf" for c in cols) else np.array([[c[i, 0]] for c in cols]))
        else:
            vals = []
            for c in cols:
                v = _row_value(c, i)
                vals.append(v.data.flat[0] if isinstance(v, Cell) and v.data.size == 1 else v)
            out_cols.append(_column_from_values(vals) if not all(
                isinstance(v, np.ndarray) and is_char(v) for v in vals) else Cell.from_list(vals, column=True))
    return TableObject(["OriginalVariableNames"] + new, out_cols)


@builtin("inner2outer")
def _inner2outer(interp, args, nargout):
    """T2 = inner2outer(T): the variables of nested tables become the outer variables."""
    t = _need_table(args, "inner2outer")
    nested = [(n, c) for n, c in zip(t.names, t.columns) if isinstance(c, TableObject)]
    if not nested:
        return t
    inner_names: list[str] = []
    for _, c in nested:
        for n in c.names:
            if n not in inner_names:
                inner_names.append(n)
    names = [n for n, c in zip(t.names, t.columns) if not isinstance(c, TableObject)]
    cols = [c for c in t.columns if not isinstance(c, TableObject)]
    for inner in inner_names:
        sub_names, sub_cols = [], []
        for outer, c in nested:
            if inner in c.names:
                sub_names.append(outer)
                sub_cols.append(c.columns[c.names.index(inner)])
        if inner in names:
            raise MatlabError(f"inner2outer: '{inner}' is both an inner variable and an outer variable.",
                              "MATLAB:table:DuplicateVarNames")
        names.append(inner)
        cols.append(TableObject(sub_names, sub_cols))
    return t.make(names, cols, t.row_names)


for _n, _f in (("addvars", _addvars), ("rows2vars", _rows2vars)):
    TableObject._METHODS = TableObject._METHODS | {_n}
    _TABLE_FUNCS[_n] = _f
    builtin(_n)(_f)


# -- sorting --------------------------------------------------------------------------------------------------------

def _sort_order(t: TableObject, keys: list[str], descend: list[bool], missing_first: bool | None = None,
                by_row_names: bool = False) -> np.ndarray:
    order = np.arange(t.nrows)
    plan = list(zip(keys, descend))
    if by_row_names:
        names = t.row_names or []
        rev = descend[0] if descend else False
        return np.array(sorted(range(len(names)), key=lambda i: names[i], reverse=rev), dtype=int)
    for key, desc in reversed(plan):
        col = t.columns[t.var_index(key)]
        ks = _col_keys(col)
        sub = [ks[i] for i in order]
        present = [i for i in range(len(sub)) if sub[i] is not _MISSING]
        absent = [i for i in range(len(sub)) if sub[i] is _MISSING]
        present.sort(key=lambda i: _row_sort_key(sub[i]), reverse=desc)
        if desc:
            # a stable descending sort keeps ties in their original order
            present = _stable_descending(sub, present)
        first = desc if missing_first is None else missing_first
        o = (absent + present) if first else (present + absent)
        order = order[np.array(o, dtype=int)] if o else order
    return order


def _row_sort_key(k: Any):
    if isinstance(k, tuple):
        return tuple(_sort_key(x) for x in k)
    return _sort_key(k)


def _stable_descending(sub: list, present: list[int]) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(present):
        j = i
        while j + 1 < len(present) and _row_sort_key(sub[present[j + 1]]) == _row_sort_key(sub[present[i]]):
            j += 1
        out += sorted(present[i:j + 1])
        i = j + 1
    return out


def _sort_spec(t: TableObject, rest: list, name: str) -> tuple[list[str], list[bool], bool, list]:
    """(sort variables, descending flags, sort by row names, remaining arguments)."""
    keys = list(t.names)
    descend: list[bool] = []
    by_rows = False
    rest = list(rest)
    if rest and not (is_text(rest[0]) and text(rest[0]).lower() in ("ascend", "descend")):
        spec = rest.pop(0)
        if is_text(spec) and text(spec) == "RowNames" or (is_text(spec) and text(spec) == t.dim_names[0]
                                                           and text(spec) not in t.names):
            by_rows = True
            keys = []
        elif is_text(spec) or isinstance(spec, (Cell, StringArray)) or getattr(spec, "is_vartype", False):
            keys = _var_names(t, spec)
        elif isinstance(spec, np.ndarray) and spec.dtype == bool:
            keys = _var_names(t, spec)
        else:
            nums = [int(k) for k in np.real(to_numeric(spec)).flatten()]
            if any(k == 0 or abs(k) > len(t.names) for k in nums):
                raise MatlabError(f"{name}: a sorting variable number is out of range.",
                                  "MATLAB:table:VarIndexOutOfRange")
            keys = [t.names[abs(k) - 1] for k in nums]
            descend = [k < 0 for k in nums]
    if rest and (is_text(rest[0]) or isinstance(rest[0], (Cell, StringArray))) and all(
            w.lower() in ("ascend", "descend") for w in (_names_from(rest[0]) if not is_text(rest[0])
                                                          else [text(rest[0])])):
        d = rest.pop(0)
        words = _names_from(d) if isinstance(d, (Cell, StringArray)) else [text(d)]
        descend = [w.lower() == "descend" for w in words]
    count = max(len(keys), 1)
    if not descend:
        descend = [False] * count
    elif len(descend) == 1:
        descend = descend * count
    elif len(descend) != len(keys):
        raise MatlabError(f"{name}: give one sort direction, or one for each sorting variable.",
                          "MATLAB:table:sortrows:WrongNumDirections")
    return keys, descend, by_rows, rest


def _sortrows(interp, args, nargout):
    """[T2, index] = sortrows(T[, vars][, direction], 'MissingPlacement', where, 'ComparisonMethod', m)."""
    t = _need_table(args, "sortrows")
    rest, opts = _options(list(args[1:]), ("MissingPlacement", "ComparisonMethod"), "sortrows")
    keys, descend, by_rows, rest = _sort_spec(t, rest, "sortrows")
    if rest:
        raise MatlabError("sortrows: unexpected extra arguments.", "MATLAB:table:sortrows:TooManyInputs")
    placement = text(opts["MissingPlacement"]).lower() if "MissingPlacement" in opts else "auto"
    if placement not in ("auto", "first", "last"):
        raise MatlabError("sortrows: 'MissingPlacement' must be 'auto', 'first' or 'last'.",
                          "MATLAB:sortrows:InvalidMissingPlacement")
    if by_rows and not t.row_names:
        return (t, np.arange(1.0, t.nrows + 1).reshape(-1, 1)) if nargout > 1 else t
    order = _sort_order(t, keys, descend, None if placement == "auto" else placement == "first", by_rows)
    if "ComparisonMethod" in opts and text(opts["ComparisonMethod"]).lower() == "abs":
        col = t.columns[t.var_index(keys[0])]
        if isinstance(col, np.ndarray) and col.dtype.kind in "iufc":
            mag = np.abs(col[:, 0].astype(complex if col.dtype.kind == "c" else float))
            order = np.argsort(-mag if descend[0] else mag, kind="stable")
    res = t.index_get(interp, [order.reshape(-1, 1) + 1.0, ix.COLON])
    return (res, (order + 1.0).reshape(-1, 1)) if nargout > 1 else res


def _topkrows(interp, args, nargout):
    """[B, index] = topkrows(T, k[, vars][, direction])."""
    t = _need_table(args, "topkrows")
    if len(args) < 2:
        raise MatlabError("topkrows needs the number of rows: topkrows(T, k).", "MATLAB:minrhs")
    k = int(np.real(to_numeric(args[1])).flat[0])
    rest, opts = _options(list(args[2:]), ("ComparisonMethod",), "topkrows")
    explicit = bool(rest) and is_text(rest[-1]) and text(rest[-1]).lower() in ("ascend", "descend") \
        or bool(rest) and isinstance(rest[-1], (Cell, StringArray)) and len(rest) > 1
    keys, descend, by_rows, _ = _sort_spec(t, rest, "topkrows")
    numeric_spec = bool(rest) and isinstance(rest[0], np.ndarray) and not is_char(rest[0]) and rest[0].dtype != bool
    if not explicit and not numeric_spec:
        descend = [True] * max(len(keys), 1)
    order = _sort_order(t, keys, descend, None, by_rows)[:max(k, 0)]
    res = t.index_get(interp, [order.reshape(-1, 1) + 1.0, ix.COLON])
    return (res, (order + 1.0).reshape(-1, 1)) if nargout > 1 else res


def _issortedrows(interp, args, nargout):
    t = _need_table(args, "issortedrows")
    keys, descend, by_rows, _ = _sort_spec(t, list(args[1:]), "issortedrows")
    order = _sort_order(t, keys, descend, None, by_rows)
    return bool_scalar(bool(np.array_equal(order, np.arange(t.nrows))))


for _n, _f in (("sortrows", _sortrows), ("topkrows", _topkrows), ("issortedrows", _issortedrows)):
    TableObject._METHODS = TableObject._METHODS | {_n}
    _TABLE_FUNCS[_n] = _f


# -- set operations on the rows of tables -----------------------------------------------------------------------------

def _table_row_keys(t: TableObject, names: list[str] | None = None) -> list[tuple]:
    names = names or t.names
    per_var = []
    for n in names:
        keys = _col_keys(t.columns[t.var_index(n)])
        per_var.append(keys)
    return [tuple(("\0missing", i) if k[i] is _MISSING else k[i] for k in per_var) for i in range(t.nrows)]


def _key_order(key: tuple):
    return tuple((2, 0) if isinstance(k, tuple) and k and k[0] == "\0missing" else _row_sort_key(k) for k in key)


def _same_vars(a: TableObject, b: TableObject, name: str) -> list[str]:
    if set(a.names) != set(b.names):
        raise MatlabError(f"{name}: the tables must have the same variable names.",
                          "MATLAB:table:setmembership:DisjointVars")
    return list(a.names)


def _unique_table(interp, args, nargout):
    """[C, ia, ic] = unique(T[, vars], 'stable' | 'sorted', 'first' | 'last')."""
    t = _need_table(args, "unique")
    words = [text(a).lower() for a in args[1:] if is_text(a) and text(a).lower() in
             ("stable", "sorted", "first", "last", "rows")]
    stable = "stable" in words
    last = "last" in words
    keys = _table_row_keys(t)
    where: dict[tuple, int] = {}
    order: list[tuple] = []
    for i, k in enumerate(keys):
        if k not in where:
            order.append(k)
            where[k] = i
        elif last:
            where[k] = i
    if not stable:
        order.sort(key=_key_order)
    ia = np.array([where[k] for k in order], dtype=int)
    rank = {k: j for j, k in enumerate(order)}
    ic = np.array([rank[k] + 1.0 for k in keys]).reshape(-1, 1)
    res = t.index_get(interp, [ia.reshape(-1, 1) + 1.0, ix.COLON])
    return (res, (ia + 1.0).reshape(-1, 1), ic)[:max(nargout, 1)] if nargout > 1 else res


def _ismember_table(interp, args, nargout):
    """[tf, loc] = ismember(A, B) for tables: which rows of A are rows of B."""
    if len(args) < 2 or not isinstance(args[0], TableObject) or not isinstance(args[1], TableObject):
        raise MatlabError("ismember: both inputs must be tables.", "MATLAB:table:setmembership:TypeMismatch")
    a, b = args[0], args[1]
    names = _same_vars(a, b, "ismember")
    where: dict[tuple, int] = {}
    for j, k in enumerate(_table_row_keys(b, names)):
        where.setdefault(k, j)
    ka = _table_row_keys(a, names)
    tf = np.array([k in where and not _has_missing(k) for k in ka], dtype=bool).reshape(-1, 1)
    loc = np.array([where[k] + 1.0 if (k in where and not _has_missing(k)) else 0.0 for k in ka]).reshape(-1, 1)
    return (tf, loc) if nargout > 1 else tf


def _has_missing(key: tuple) -> bool:
    return any(isinstance(k, tuple) and k and k[0] == "\0missing" for k in key)


def _set_op(name: str):
    def f(interp, args, nargout):
        if len(args) < 2 or not isinstance(args[0], TableObject) or not isinstance(args[1], TableObject):
            raise MatlabError(f"{name}: both inputs must be tables.", "MATLAB:table:setmembership:TypeMismatch")
        a, b = args[0], args[1]
        names = _same_vars(a, b, name)
        stable = any(is_text(x) and text(x).lower() == "stable" for x in args[2:])
        ka, kb = _table_row_keys(a, names), _table_row_keys(b, names)
        sa, sb = set(ka), set(kb)
        pick: list[tuple[int, int]] = []  # (which table, row)
        seen: set = set()

        def take(keys, which, cond):
            for i, k in enumerate(keys):
                if k not in seen and cond(k):
                    seen.add(k)
                    pick.append((which, i))
        if name == "intersect":
            take(ka, 0, lambda k: k in sb and not _has_missing(k))
        elif name == "setdiff":
            take(ka, 0, lambda k: k not in sb or _has_missing(k))
        elif name == "union":
            take(ka, 0, lambda k: True)
            take(kb, 1, lambda k: True)
        else:  # setxor
            take(ka, 0, lambda k: k not in sb or _has_missing(k))
            take(kb, 1, lambda k: k not in sa or _has_missing(k))
        if not stable:
            pick.sort(key=lambda p: _key_order((ka, kb)[p[0]][p[1]]))
        b_sel = b.index_get(interp, [ix.COLON, Cell.from_list([char(n) for n in names])])
        parts = []
        for which, i in pick:
            src = a if which == 0 else b_sel
            parts.append(src.index_get(interp, [np.array([[i + 1.0]]), ix.COLON]))
        if parts:
            for p in parts:
                p.row_names = None
            res = TableObject.concat(parts, 0) if len(parts) > 1 else parts[0]
        else:
            res = a.index_get(interp, [np.zeros((0, 1)), ix.COLON])
        ia = np.array([i + 1.0 for which, i in pick if which == 0]).reshape(-1, 1)
        ib = np.array([i + 1.0 for which, i in pick if which == 1]).reshape(-1, 1)
        if name == "intersect":
            first_b: dict[tuple, int] = {}
            for j, k in enumerate(kb):
                first_b.setdefault(k, j)
            ib = np.array([first_b[ka[i]] + 1.0 for _w, i in pick]).reshape(-1, 1)
        return (res, ia, ib)[:max(nargout, 1)] if nargout > 1 else res
    return f


for _n, _f in (("unique", _unique_table), ("ismember", _ismember_table), ("intersect", _set_op("intersect")),
               ("union", _set_op("union")), ("setdiff", _set_op("setdiff")), ("setxor", _set_op("setxor"))):
    TableObject._METHODS = TableObject._METHODS | {_n}
    _TABLE_FUNCS[_n] = _f


# -- missing data in tables -------------------------------------------------------------------------------------------

def _ismissing_table(interp, args, nargout):
    """TF = ismissing(T[, indicators]): a logical matrix with one column per variable."""
    t = _need_table(args, "ismissing")
    cols = []
    for c in t.columns:
        if isinstance(c, TableObject):
            m = _ismissing_table(interp, [c, *args[1:]], 1)
            cols.append(m.any(axis=1, keepdims=True))
            continue
        if len(args) > 1:
            m = np.asarray(BUILTINS["ismissing"](interp, [c, args[1]], 1), dtype=bool)
        elif _is_time(c) or isinstance(c, (Cell, StringArray)) or isinstance(c, np.ndarray):
            m = _missing_mask(c) if not is_char(c) else np.all(c == " ", axis=1, keepdims=True)
        else:
            m = np.zeros((t.nrows, 1), dtype=bool)
        m = np.asarray(m, dtype=bool).reshape(t.nrows, -1)
        cols.append(m.any(axis=1, keepdims=True))
    return np.hstack(cols) if cols else np.zeros((t.nrows, 0), dtype=bool)


def _standardize_table(interp, args, nargout):
    """T2 = standardizeMissing(T, indicators, 'DataVariables', vars)."""
    t = _need_table(args, "standardizeMissing")
    rest, opts = _options(list(args[1:]), ("DataVariables", "ReplaceValues"), "standardizeMissing", 1)
    if not rest:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    targets = _var_names(t, opts["DataVariables"]) if "DataVariables" in opts else list(t.names)
    out = t.copy()
    plain = BUILTINS["standardizeMissing"]
    for n in targets:
        k = out.names.index(n)
        col = out.columns[k]
        try:
            out.columns[k] = _as_column(plain(interp, [col, rest[0]], 1)) if not isinstance(col, TableObject) else col
        except MatlabError:
            continue  # indicators of another type do not apply to this variable
    return out


for _n, _f in (("ismissing", _ismissing_table), ("standardizeMissing", _standardize_table)):
    TableObject._METHODS = TableObject._METHODS | {_n}
    _TABLE_FUNCS[_n] = _f


# -- summary --------------------------------------------------------------------------------------------------------

def _class_text(c: Any) -> str:
    if isinstance(c, Cell):
        if c.data.size and all(is_char(x) for x in c.data.flat):
            return "cell array of character vectors"
        return "cell"
    if getattr(c, "kind", "") == "categorical" and getattr(c, "ordinal", False):
        return "ordinal categorical"
    return class_name(c)


def _shape_text(c: Any) -> str:
    from ..values import shape_of
    return "×".join(str(d) for d in shape_of(c))


def _num_text(x: float) -> str:
    if x != x:
        return "NaN"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.5g}"


def _summary_info(interp, t: TableObject) -> list[dict]:
    info = []
    for n, c in zip(t.names, t.columns):
        entry: dict[str, Any] = {"name": n, "size": _shape_text(c), "type": _class_text(c),
                                 "units": t.units.get(n, ""), "description": t.var_descriptions.get(n, ""),
                                 "values": []}
        miss = _missing_mask(c) if not isinstance(c, TableObject) and not is_char(c) and not isinstance(c, Struct) \
            else np.zeros((t.nrows, 1), dtype=bool)
        entry["nummissing"] = int(np.sum(miss))
        if isinstance(c, np.ndarray) and c.dtype == bool:
            entry["values"] = [("True", str(int(np.sum(c)))), ("False", str(int(np.sum(~c))))]
        elif isinstance(c, np.ndarray) and c.dtype.kind in "iuf" and c.size:
            a = c.astype(float)
            if c.shape[1] == 1 and np.any(~np.isnan(a)):
                entry["values"] = [("Min", _num_text(float(np.nanmin(a)))), ("Median", _num_text(float(
                    np.nanmedian(a)))), ("Max", _num_text(float(np.nanmax(a))))]
                entry["stats"] = (float(np.nanmin(a)), float(np.nanmedian(a)), float(np.nanmax(a)))
        elif getattr(c, "kind", "") in ("datetime", "duration") and c.data.size and np.any(~np.isnan(c.data)):
            lo, hi = float(np.nanmin(c.data)), float(np.nanmax(c.data))
            med = float(np.nanmedian(c.data))
            texts = [str(c.like(np.array([[v]])).element_texts().flat[0]) for v in (lo, med, hi)]
            entry["values"] = list(zip(("Min", "Median", "Max"), texts))
            entry["time_stats"] = (c.like(np.array([[lo]])), c.like(np.array([[med]])), c.like(np.array([[hi]])))
        elif getattr(c, "kind", "") == "categorical":
            counts = [(cat, str(int(np.sum(c.data == k + 1)))) for k, cat in enumerate(c.categories)]
            und = int(np.sum(np.isnan(c.data)))
            entry["values"] = counts + ([("<undefined>", str(und))] if und else [])
        info.append(entry)
    return info


def _summary(interp, args, nargout):
    """summary(T) prints a description of every variable; s = summary(T) returns it as a struct."""
    t = _need_table(args, "summary")
    info = _summary_info(interp, t)
    if nargout > 0:
        out = Struct()
        for e in info:
            s = Struct.scalar({"Size": np.array([[float(x) for x in e["size"].split("×")]]),
                               "Type": char(e["type"]), "Description": char(e["description"]),
                               "Units": char(e["units"]), "Continuity": np.zeros((0, 0))})
            if "stats" in e:
                for key, v in zip(("Min", "Median", "Max"), e["stats"]):
                    s.set_field(0, key, np.array([[v]]))
                s.set_field(0, "NumMissing", np.array([[float(e["nummissing"])]]))
            elif "time_stats" in e:
                for key, v in zip(("Min", "Median", "Max"), e["time_stats"]):
                    s.set_field(0, key, v)
                s.set_field(0, "NumMissing", np.array([[float(e["nummissing"])]]))
            out.set_field(0, e["name"], s)
        return out
    lines: list[str] = []
    if t.description:
        lines += [f"Description:  {t.description}", ""]
    row_times = getattr(t, "row_times", None)
    if row_times is not None:
        lines += ["RowTimes:", ""]
        lines.append(f"    {t.dim_names[0]}: {t.nrows}×1 {row_times.class_name()}")
        if row_times.data.size and np.any(~np.isnan(row_times.data)):
            lo, hi = float(np.nanmin(row_times.data)), float(np.nanmax(row_times.data))
            med = float(np.nanmedian(row_times.data))
            texts = [str(row_times.like(np.array([[v]])).element_texts().flat[0]) for v in (lo, med, hi)]
            lines += ["        Values:"] + [f"            {k.ljust(10)}{v}" for k, v in
                                           zip(("Min", "Median", "Max"), texts)]
        lines.append("")
    lines += ["Variables:", ""]
    for e in info:
        lines.append(f"    {e['name']}: {e['size']} {e['type']}")
        lines.append("")
        props = [(k, e[k.lower()]) for k in ("Units", "Description") if e[k.lower()]]
        if props:
            lines.append("        Properties:")
            lines += [f"            {(k + ':').ljust(13)}{v}" for k, v in props]
        if e["values"]:
            lines.append("        Values:")
            lines.append("")
            width = max(len(k) for k, _ in e["values"])
            vwidth = max(len(v) for _, v in e["values"])
            lines += [f"            {k.ljust(width)}    {v.rjust(vwidth)}" for k, v in e["values"]]
            if e["nummissing"] and e["type"] not in ("categorical", "ordinal categorical", "logical"):
                lines.append(f"            {'NumMissing'.ljust(width)}    {str(e['nummissing']).rjust(vwidth)}")
            lines.append("")
    interp.out("\n".join(lines) + "\n")
    return None


TableObject._METHODS = TableObject._METHODS | {"summary"}
_TABLE_FUNCS["summary"] = _summary


# -- conversion options ----------------------------------------------------------------------------------

def _row_names_from(opts: dict) -> list[str] | None:
    if "RowNames" not in opts:
        return None
    v = opts["RowNames"]
    if (isinstance(v, np.ndarray) and v.size == 0) or (isinstance(v, Cell) and v.data.size == 0):
        return None
    return _names_from(v)


def _apply_dim_names(t: TableObject, opts: dict) -> TableObject:
    if "DimensionNames" in opts:
        t.apply_properties(Struct.scalar({"DimensionNames": opts["DimensionNames"]}))
    return t


@builtin("array2table")
def _array2table(interp, args, nargout):
    """T = array2table(A, 'VariableNames', names, 'RowNames', names, 'DimensionNames', names)."""
    rest, opts = _options(list(args), ("VariableNames", "RowNames", "DimensionNames"), "array2table", 1)
    nargs(rest, 1, 1, "array2table")
    a = rest[0]
    if isinstance(a, (Cell, StringArray)):
        data = a.data
        ncols = data.shape[1] if data.ndim == 2 else 1
        parts = [type(a)(data[:, k:k + 1].copy()) for k in range(ncols)]
    elif _is_time(a):
        ncols = a.data.shape[1]
        parts = [a.like(a.data[:, k:k + 1].copy()) for k in range(ncols)]
    elif isinstance(a, np.ndarray) and a.ndim == 2:
        ncols = a.shape[1]
        parts = [a[:, k:k + 1].copy() for k in range(ncols)]
    else:
        raise MatlabError("array2table: the input must be a 2-D array.", "MATLAB:array2table:NDArray")
    arg_names = interp.state.get("call_arg_names") or []
    base = arg_names[0] if arg_names and arg_names[0] else "Var"
    if "VariableNames" in opts:
        names = _names_from(opts["VariableNames"])
        if len(names) != ncols:
            raise MatlabError("The VariableNames property must contain one name for each variable in the table.",
                              "MATLAB:table:IncorrectNumberOfVarNames")
    else:
        names = [f"{base}{k + 1}" for k in range(ncols)]
    return _apply_dim_names(TableObject(names, parts, row_names=_row_names_from(opts)), opts)


@builtin("cell2table")
def _cell2table(interp, args, nargout):
    """T = cell2table(C, 'VariableNames', names, 'RowNames', names, 'DimensionNames', names)."""
    rest, opts = _options(list(args), ("VariableNames", "RowNames", "DimensionNames"), "cell2table", 1)
    nargs(rest, 1, 1, "cell2table")
    c = rest[0]
    if not isinstance(c, Cell) or c.data.ndim != 2:
        raise MatlabError("cell2table: the input must be a 2-D cell array.", "MATLAB:cell2table:NDCell")
    nrows, ncols = c.data.shape
    arg_names = interp.state.get("call_arg_names") or []
    base = arg_names[0] if arg_names and arg_names[0] else "Var"
    names = _names_from(opts["VariableNames"]) if "VariableNames" in opts else [f"{base}{k + 1}" for k in
                                                                                 range(ncols)]
    if len(names) != ncols:
        raise MatlabError("The VariableNames property must contain one name for each variable in the table.",
                          "MATLAB:table:IncorrectNumberOfVarNames")
    cols = []
    for k in range(ncols):
        vals = [c.data[i, k] for i in range(nrows)]
        col = _column_from_values(vals)
        if isinstance(col, Cell) and vals and all(isinstance(v, np.ndarray) and not is_char(v) and v.size == 1
                                                  and v.dtype.kind in "biuf" for v in vals):
            col = np.array([[float(v.flat[0])] for v in vals])  # mixed numeric classes: double
        cols.append(col)
    return _apply_dim_names(TableObject(names, cols, row_names=_row_names_from(opts)), opts)


@builtin("struct2table")
def _struct2table(interp, args, nargout):
    """T = struct2table(S, 'AsArray', tf, 'RowNames', names, 'DimensionNames', names)."""
    rest, opts = _options(list(args), ("AsArray", "RowNames", "DimensionNames"), "struct2table", 1)
    nargs(rest, 1, 1, "struct2table")
    s = rest[0]
    if not isinstance(s, Struct):
        raise MatlabError("struct2table: the input must be a struct.", "MATLAB:struct2table:NonStructInput")
    as_array = _flag(opts["AsArray"]) if "AsArray" in opts else s.numel() != 1
    if not as_array:
        d = s.data.flat[0]
        cols = [_as_column(d[f]) for f in s.fields]
        if len({_height(c) for c in cols}) > 1:
            raise MatlabError("struct2table: the fields of a scalar struct must have the same number of rows; "
                              "use 'AsArray', true to make a one-row table.",
                              "MATLAB:struct2table:UnequalFieldLengths")
        return _apply_dim_names(TableObject(list(s.fields), cols, row_names=_row_names_from(opts)), opts)
    elems = list(s.data.flatten(order="F"))
    cols = []
    for f in s.fields:
        vals = [e[f] for e in elems]
        col = _column_from_values(vals)
        if isinstance(col, Cell) and len(vals) == 1:
            pass
        cols.append(col)
    return _apply_dim_names(TableObject(list(s.fields), cols, row_names=_row_names_from(opts)), opts)


def _table2struct(interp, args, nargout):
    """S = table2struct(T) (one element per row) or table2struct(T, 'ToScalar', true)."""
    t = _need_table(args, "table2struct")
    _, opts = _options(list(args[1:]), ("ToScalar",), "table2struct")
    if "ToScalar" in opts and _flag(opts["ToScalar"]):
        return Struct.scalar(dict(zip(t.names, t.columns)))
    data = np.empty((t.nrows, 1), dtype=object)
    for i in range(t.nrows):
        elem = {}
        for n, c in zip(t.names, t.columns):
            v = _row_value(c, i)
            elem[n] = v.data.flat[0] if isinstance(v, Cell) and v.data.size == 1 else v
        data[i, 0] = elem
    return Struct(data, list(t.names))


TableObject._METHODS = TableObject._METHODS | {"table2struct"}
_TABLE_FUNCS["table2struct"] = _table2struct
builtin("table2struct")(_table2struct)


# -- custom properties -------------------------------------------------------------------------------------------------

def _addprop(interp, args, nargout):
    """T = addprop(T, names, types): add custom properties ('table' or 'variable' level)."""
    t = _need_table(args, "addprop")
    nargs(args, 3, 3, "addprop")
    names, kinds = _names_from(args[1]), [k.lower() for k in _names_from(args[2])]
    if len(kinds) == 1:
        kinds = kinds * len(names)
    if len(kinds) != len(names):
        raise MatlabError("addprop: give one type ('table' or 'variable') for each property name.",
                          "MATLAB:table:addprop:WrongNumTypes")
    out = t.copy()
    for n, k in zip(names, kinds):
        if k not in ("table", "variable"):
            raise MatlabError("addprop: the property type must be 'table' or 'variable'.",
                              "MATLAB:table:addprop:InvalidType")
        if n in out.custom:
            raise MatlabError(f"addprop: the table already has a custom property '{n}'.",
                              "MATLAB:table:addprop:DuplicateProp")
        out.custom[n] = np.zeros((0, 0))
        if k == "variable":
            out.custom_var.add(n)
    return out


def _rmprop(interp, args, nargout):
    """T = rmprop(T, names): remove custom properties."""
    t = _need_table(args, "rmprop")
    nargs(args, 2, 2, "rmprop")
    out = t.copy()
    for n in _names_from(args[1]):
        if n not in out.custom:
            raise MatlabError(f"rmprop: '{n}' is not a custom property of this table.",
                              "MATLAB:table:rmprop:UnknownProp")
        del out.custom[n]
        out.custom_var.discard(n)
    return out


for _n, _f in (("addprop", _addprop), ("rmprop", _rmprop)):
    TableObject._METHODS = TableObject._METHODS | {_n}
    _TABLE_FUNCS[_n] = _f
