"""The table type: named columns of equal height."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from .. import indexing as ix
from ..display import format_scalar, size_str, summary
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, MString, StringArray, Struct, char, is_char, string_data, string_result, to_str,
)
from ._registry import builtin
from ._util import bool_scalar, is_text, num, text


def _height(v: Any) -> int:
    if isinstance(v, (Cell, StringArray)):
        return v.data.shape[0]
    if isinstance(v, np.ndarray):
        return v.shape[0]
    if isinstance(v, MString):
        return 1
    raise MatlabError(f"A {type(v).__name__} cannot be a table variable.")


def _as_column(v: Any) -> Any:
    """Table variables are stored with one row per table row (row vectors become columns)."""
    if isinstance(v, MString):
        d = np.empty((1, 1), dtype=object)
        d[0, 0] = str(v)
        return StringArray(d)
    if isinstance(v, Cell) and v.data.ndim == 2 and v.data.shape[0] == 1 and v.data.shape[1] > 1:
        return Cell(v.data.T.copy())
    if isinstance(v, StringArray) and v.data.shape[0] == 1 and v.data.shape[1] > 1:
        return StringArray(v.data.T.copy())
    if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] > 1 and not is_char(v):
        return v.T.copy()
    return v


def _rows(v: Any, pos: np.ndarray) -> Any:
    if isinstance(v, Cell):
        return Cell(v.data[pos, ...])
    if isinstance(v, StringArray):
        return StringArray(v.data[pos, ...])
    return v[pos, ...]


def _names_from(v: Any) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x) for x in v.values_f()]
    if isinstance(v, StringArray):
        return [x for x in v.values_f()]
    return [text(v)]


class TableObject(MObject):
    def __init__(self, names: list[str], columns: list[Any], description: str = ""):
        if len(set(names)) != len(names):
            raise MatlabError("Duplicate table variable names.")
        heights = {_height(c) for c in columns}
        if len(heights) > 1:
            raise MatlabError("All table variables must have the same number of rows.")
        self.names = list(names)
        self.columns = list(columns)
        self.description = description

    @property
    def nrows(self) -> int:
        return _height(self.columns[0]) if self.columns else 0

    def class_name(self) -> str:
        return "table"

    def matlab_shape(self) -> tuple:
        return (self.nrows, len(self.names))

    def copy(self) -> "TableObject":
        return TableObject(list(self.names), list(self.columns), self.description)

    # -- lookup helpers --------------------------------------------------------------------
    def var_index(self, name: str) -> int:
        if name not in self.names:
            raise MatlabError(f"Unrecognized table variable name '{name}'.")
        return self.names.index(name)

    def var_positions(self, idx: Any) -> list[int]:
        if idx is ix.COLON:
            return list(range(len(self.names)))
        if is_text(idx) or isinstance(idx, (Cell, StringArray)):
            return [self.var_index(n) for n in _names_from(idx)]
        pos, _ = ix._to_positions(idx, len(self.names), for_set=False)
        return [int(p) for p in pos]

    def row_positions(self, idx: Any, for_set: bool = False) -> np.ndarray:
        pos, _ = ix._to_positions(idx, self.nrows, for_set=for_set)
        return pos

    # -- protocol ----------------------------------------------------------------------------
    _METHODS = {"height", "width", "head", "tail", "summary", "sortrows", "removevars", "addvars",
                "renamevars", "table2array", "table2struct", "istable", "varfun", "rows2vars"}

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        from ..values import to_matlab
        fn = _TABLE_FUNCS[name]
        res = fn(interp, args, nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    def get_field(self, interp, name: str) -> Any:
        if name == "Properties":
            return Struct.scalar({
                "VariableNames": Cell.from_list([char(n) for n in self.names]),
                "Description": char(self.description),
                "RowNames": Cell(np.empty((0, 1), dtype=object)),
            })
        if name in self.names:
            return self.columns[self.names.index(name)]
        raise MatlabError(f"Unrecognized table variable name '{name}'.")

    def set_field(self, interp, name: str, value: Any) -> "TableObject":
        t = self.copy()
        if name == "Properties":
            if not isinstance(value, Struct):
                raise MatlabError("Properties must be a struct.")
            new_names = _names_from(value.get_field(0, "VariableNames"))
            if len(new_names) != len(t.names):
                raise MatlabError("The number of variable names must match the number of variables.")
            t.names = new_names
            if "Description" in value.fields:
                t.description = to_str(value.get_field(0, "Description"))
            return t
        col = _as_column(value)
        if t.names and _height(col) != t.nrows:
            if isinstance(col, np.ndarray) and col.size == 1:
                col = np.repeat(col, t.nrows, axis=0)
            else:
                raise MatlabError(f"To assign to or create a variable in a table, the number of rows must "
                                  f"match the height of the table ({t.nrows}).")
        if name in t.names:
            t.columns[t.names.index(name)] = col
        else:
            t.names.append(name)
            t.columns.append(col)
        return t

    def index_get(self, interp, args: list) -> Any:
        if len(args) != 2:
            raise MatlabError("Tables are indexed with two subscripts: T(rows, vars).")
        rows = self.row_positions(args[0])
        cols = self.var_positions(args[1])
        return TableObject([self.names[c] for c in cols], [_rows(self.columns[c], rows) for c in cols],
                           self.description)

    def brace_get(self, interp, args: list) -> list:
        if len(args) != 2:
            raise MatlabError("Tables are indexed with two subscripts: T{rows, vars}.")
        sub = self.index_get(interp, args)
        return [sub.to_array()]

    def index_delete(self, interp, args: list) -> Any:
        if len(args) != 2:
            raise MatlabError("Use T(rows, :) = [] or T(:, vars) = [] to delete from a table.")
        rows_all = args[0] is ix.COLON
        if rows_all:
            drop = set(self.var_positions(args[1]))
            keep = [k for k in range(len(self.names)) if k not in drop]
            return TableObject([self.names[k] for k in keep], [self.columns[k] for k in keep], self.description)
        if args[1] is not ix.COLON:
            raise MatlabError("Delete whole rows with T(rows, :) = [].")
        keep = np.ones(self.nrows, dtype=bool)
        keep[self.row_positions(args[0])] = False
        idx = np.nonzero(keep)[0]
        return TableObject(list(self.names), [_rows(c, idx) for c in self.columns], self.description)

    def index_set(self, interp, args: list, value: Any) -> Any:
        if len(args) != 2:
            raise MatlabError("Tables are indexed with two subscripts: T(rows, vars) = value.")
        rows = self.row_positions(args[0], for_set=True) if args[0] is not ix.COLON else np.arange(self.nrows)
        cols = self.var_positions(args[1])
        if isinstance(value, TableObject):
            pieces = value.columns
        elif isinstance(value, Cell):
            if value.data.shape[1] != len(cols):
                raise MatlabError("The cell array must have one column per table variable.")
            pieces = []
            for j in range(len(cols)):
                col_vals = [value.data[i, j] for i in range(value.data.shape[0])]
                pieces.append(self._column_like(self.columns[cols[j]], col_vals))
        else:
            raise MatlabError("Assign a table or a cell array to table rows.")
        t = self.copy()
        for j, c in enumerate(cols):
            t.columns[c] = _assign_rows(t.columns[c], rows, pieces[j])
        n = max(_height(c) for c in t.columns)
        t.columns = [_pad_rows(c, n) for c in t.columns]
        return t

    @staticmethod
    def _column_like(template: Any, vals: list) -> Any:
        if isinstance(template, Cell):
            d = np.empty((len(vals), 1), dtype=object)
            for i, v in enumerate(vals):
                d[i, 0] = v
            return Cell(d)
        if isinstance(template, StringArray):
            d = np.empty((len(vals), 1), dtype=object)
            for i, v in enumerate(vals):
                d[i, 0] = to_str(v)
            return StringArray(d)
        return np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in vals])

    def to_array(self) -> Any:
        from ..ops import concat
        if not self.columns:
            return np.zeros((0, 0))
        return concat([c if not isinstance(c, StringArray) else c for c in self.columns], 1)

    @staticmethod
    def concat(items: list, axis: int) -> "TableObject":
        tables = [t for t in items if isinstance(t, TableObject)]
        if len(tables) != len([t for t in items if not (isinstance(t, np.ndarray) and t.size == 0)]):
            raise MatlabError("Tables can only be concatenated with other tables.")
        from ..ops import concat as cat
        if axis == 1:
            names, cols = [], []
            for t in tables:
                names += t.names
                cols += t.columns
            return TableObject(names, cols, tables[0].description)
        first = tables[0]
        for t in tables[1:]:
            if set(t.names) != set(first.names):
                raise MatlabError("Vertically concatenated tables must have the same variable names.")
        cols = [cat([t.columns[t.names.index(n)] for t in tables], 0) for n in first.names]
        return TableObject(list(first.names), cols, first.description)

    # -- display -------------------------------------------------------------------------------
    def _cell_text(self, col: Any, i: int) -> str:
        if isinstance(col, Cell):
            return "{" + summary(col.data[i, 0], True) + "}"
        if isinstance(col, StringArray):
            x = col.data[i, 0]
            return "<missing>" if x is None else f'"{x}"'
        if is_char(col):
            return "'" + "".join(col[i]) + "'"
        row = col[i]
        if col.dtype == bool:
            return " ".join("true" if x else "false" for x in row)
        if col.dtype.kind == "f":
            finite = col[np.isfinite(col)]
            if finite.size and not np.all(finite == np.round(finite)):
                return " ".join(f"{x:.4f}" if np.isfinite(x) else format_scalar(x) for x in row)
        return " ".join(format_scalar(x) for x in row)

    def display_lines(self) -> list[str]:
        if not self.names:
            return [f"  {size_str(self.matlab_shape())} empty table"]
        limit = 20
        n = self.nrows
        shown = list(range(n)) if n <= limit else list(range(10)) + list(range(n - 5, n))
        cells = [[self._cell_text(c, i) for c in self.columns] for i in shown]
        widths = [max([len(self.names[j])] + [len(r[j]) for r in cells]) for j in range(len(self.names))]
        lines = [f"  {size_str(self.matlab_shape())} table", ""]
        lines.append("    " + "    ".join(self.names[j].center(widths[j]) for j in range(len(widths))).rstrip())
        lines.append("    " + "    ".join("_" * widths[j] for j in range(len(widths))))
        lines.append("")
        for k, r in enumerate(cells):
            if n > limit and k == 10:
                lines.append("    " + ":".center(sum(widths) + 4 * (len(widths) - 1)))
            lines.append("    " + "    ".join(r[j].center(widths[j]) if not r[j][:1].isdigit() and
                                                r[j][:1] != "-" else r[j].rjust(widths[j])
                                                for j in range(len(widths))).rstrip())
        return lines

    def format_display(self, name: str) -> str:
        return f"{name} =\n\n" + "\n".join(self.display_lines()) + "\n\n"


def _assign_rows(col: Any, rows: np.ndarray, piece: Any) -> Any:
    if isinstance(col, Cell):
        data = ix.set_(col.data, [rows.reshape(-1, 1) + 1.0, ix.COLON], piece.data if isinstance(piece, Cell)
                       else _obj(piece), lambda: np.zeros((0, 0)))
        return Cell(data)
    if isinstance(col, StringArray):
        return StringArray(ix.set_(col.data, [rows.reshape(-1, 1) + 1.0, ix.COLON], string_data(piece),
                                   lambda: None))
    vals = num(piece) if not isinstance(piece, np.ndarray) else piece
    return ix.set_(col, [rows.reshape(-1, 1) + 1.0, ix.COLON], vals, lambda: 0)


def _obj(v: Any) -> np.ndarray:
    d = np.empty((1, 1), dtype=object)
    d[0, 0] = v
    return d


def _pad_rows(col: Any, n: int) -> Any:
    h = _height(col)
    if h >= n:
        return col
    extra = n - h
    if isinstance(col, Cell):
        pad = np.empty((extra,) + col.data.shape[1:], dtype=object)
        for idx in np.ndindex(pad.shape):
            pad[idx] = np.zeros((0, 0))
        return Cell(np.concatenate([col.data, pad]))
    if isinstance(col, StringArray):
        pad = np.empty((extra,) + col.data.shape[1:], dtype=object)
        return StringArray(np.concatenate([col.data, pad]))
    fill = np.nan if col.dtype.kind == "f" else 0
    return np.concatenate([col, np.full((extra,) + col.shape[1:], fill, dtype=col.dtype)])


# -- constructors and conversions ----------------------------------------------------------------

def _split_opts(args: list, names: set[str]) -> tuple[list, dict]:
    opts, rest, i = {}, [], 0
    while i < len(args):
        a = args[i]
        if is_text(a) and text(a) in names and i + 1 < len(args):
            opts[text(a)] = args[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    return rest, opts


def _table(interp, args: list, nargout: int):
    rest, opts = _split_opts(args, {"VariableNames", "RowNames", "Size", "VariableTypes"})
    if "Size" in opts:
        h, w = (int(x) for x in num(opts["Size"]).flatten()[:2])
        types = _names_from(opts["VariableTypes"]) if "VariableTypes" in opts else ["double"] * w
        names = _names_from(opts["VariableNames"]) if "VariableNames" in opts else [f"Var{k + 1}" for k in range(w)]
        cols = []
        for t in types:
            if t in ("cell", "cellstr"):
                cols.append(Cell.empty(h, 1))
            elif t == "string":
                d = np.empty((h, 1), dtype=object)
                cols.append(StringArray(d))
            elif t == "logical":
                cols.append(np.zeros((h, 1), dtype=bool))
            else:
                cols.append(np.zeros((h, 1)))
        return TableObject(names, cols)
    cols = [_as_column(a) for a in rest]
    arg_names = interp.state.get("call_arg_names") or []
    if "VariableNames" in opts:
        names = _names_from(opts["VariableNames"])
    else:
        names = [arg_names[k] if k < len(arg_names) and arg_names[k] else f"Var{k + 1}" for k in range(len(cols))]
    if len(names) != len(cols):
        raise MatlabError("The number of variable names must match the number of variables.")
    return TableObject(names, cols)


builtin("table")(_table)


@builtin("array2table")
def _array2table(interp, args, nargout):
    rest, opts = _split_opts(args, {"VariableNames", "RowNames"})
    a = rest[0]
    n = a.shape[1]
    names = _names_from(opts["VariableNames"]) if "VariableNames" in opts else [f"Var{k + 1}" for k in range(n)]
    return TableObject(names, [a[:, k:k + 1].copy() for k in range(n)])


@builtin("cell2table")
def _cell2table(interp, args, nargout):
    rest, opts = _split_opts(args, {"VariableNames", "RowNames"})
    c = rest[0]
    n = c.data.shape[1]
    names = _names_from(opts["VariableNames"]) if "VariableNames" in opts else [f"Var{k + 1}" for k in range(n)]
    cols = []
    for k in range(n):
        vals = [c.data[i, k] for i in range(c.data.shape[0])]
        if vals and all(isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.size == 1 for v in vals):
            cols.append(np.array([[float(v.flat[0])] for v in vals]))
        else:
            d = np.empty((len(vals), 1), dtype=object)
            for i, v in enumerate(vals):
                d[i, 0] = v
            cols.append(Cell(d))
    return TableObject(names, cols)


@builtin("struct2table")
def _struct2table(interp, args, nargout):
    s = args[0]
    if s.numel() == 1:
        d = s.data.flat[0]
        return TableObject(list(s.fields), [_as_column(d[f]) for f in s.fields])
    elems = list(s.data.flatten(order="F"))
    cols = []
    for f in s.fields:
        vals = [e[f] for e in elems]
        cols.append(TableObject._column_like(Cell(np.empty((0, 1), dtype=object)), vals)
                    if not all(isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.size == 1 for v in vals)
                    else np.array([[float(v.flat[0])] for v in vals]))
    return TableObject(list(s.fields), cols)


def _table2struct(interp, args, nargout):
    t = args[0]
    data = np.empty((t.nrows, 1), dtype=object)
    for i in range(t.nrows):
        elem = {}
        for n, c in zip(t.names, t.columns):
            if isinstance(c, Cell):
                elem[n] = c.data[i, 0]
            elif isinstance(c, StringArray):
                elem[n] = string_result(c.data[i:i + 1, :])
            else:
                elem[n] = c[i:i + 1, :]
        data[i, 0] = elem
    return Struct(data, list(t.names))


def _table2array(interp, args, nargout):
    return args[0].to_array()


def _height_fn(interp, args, nargout):
    return float(args[0].nrows)


def _width_fn(interp, args, nargout):
    return float(len(args[0].names))


def _head(interp, args, nargout):
    t = args[0]
    k = int(num(args[1]).flat[0]) if len(args) > 1 else 8
    return t.index_get(interp, [np.arange(1, min(k, t.nrows) + 1, dtype=float).reshape(1, -1), ix.COLON])


def _tail(interp, args, nargout):
    t = args[0]
    k = int(num(args[1]).flat[0]) if len(args) > 1 else 8
    start = max(t.nrows - k, 0)
    return t.index_get(interp, [np.arange(start + 1, t.nrows + 1, dtype=float).reshape(1, -1), ix.COLON])


def _sortrows(interp, args, nargout):
    t = args[0]
    keys = t.names[:1]
    descend = False
    for a in args[1:]:
        if is_text(a) and text(a).lower() in ("ascend", "descend"):
            descend = text(a).lower() == "descend"
        elif is_text(a) or isinstance(a, (Cell, StringArray)):
            keys = _names_from(a)
        else:
            keys = [t.names[int(k) - 1] for k in num(a).flatten()]
    order = np.arange(t.nrows)
    for key in reversed(keys):
        col = t.columns[t.var_index(key)]
        if isinstance(col, (Cell, StringArray)):
            vals = [to_str(x) if isinstance(col, Cell) else (x or "") for x in col.data[order, 0]]
            o = sorted(range(len(vals)), key=lambda i: vals[i], reverse=descend)
        else:
            v = col[order, 0].astype(float)
            o = list(np.argsort(-v if descend else v, kind="stable"))
        order = order[np.array(o, dtype=int)]
    res = t.index_get(interp, [order.reshape(1, -1) + 1.0, ix.COLON])
    return (res, (order + 1.0).reshape(-1, 1)) if nargout > 1 else res


def _removevars(interp, args, nargout):
    return args[0].index_delete(interp, [ix.COLON, args[1]])


def _addvars(interp, args, nargout):
    t = args[0]
    rest, opts = _split_opts(list(args[1:]), {"NewVariableNames", "Before", "After"})
    names = _names_from(opts["NewVariableNames"]) if "NewVariableNames" in opts else \
        [f"Var{len(t.names) + k + 1}" for k in range(len(rest))]
    for n, v in zip(names, rest):
        t = t.set_field(interp, n, v)
    return t


def _renamevars(interp, args, nargout):
    t = args[0].copy()
    old, new = _names_from(args[1]), _names_from(args[2])
    for o, n in zip(old, new):
        t.names[t.var_index(o)] = n
    return t


def _istable(interp, args, nargout):
    return bool_scalar(isinstance(args[0], TableObject))


def _summary(interp, args, nargout):
    t = args[0]
    lines = [f"Variables ({t.nrows} rows):", ""]
    for n, c in zip(t.names, t.columns):
        if isinstance(c, np.ndarray) and c.dtype.kind in "biuf" and c.size:
            v = c.astype(float)
            lines.append(f"    {n}: {c.shape[0]}x{c.shape[1]} {c.dtype.name if c.dtype.kind != 'f' else 'double'}")
            lines.append(f"        Min {np.nanmin(v):g}   Median {np.nanmedian(v):g}   Max {np.nanmax(v):g}")
        else:
            kind = "cell" if isinstance(c, Cell) else ("string" if isinstance(c, StringArray) else "char")
            lines.append(f"    {n}: {_height(c)}x1 {kind}")
    interp.out("\n".join(lines) + "\n")
    return None


def _varfun(interp, args, nargout):
    fn, t = args[0], args[1]
    fname = fn.name if getattr(fn, "name", None) else "Fun"
    names, cols = [], []
    for n, c in zip(t.names, t.columns):
        res = interp.call_function_value(fn, [c], 1)
        names.append(f"{fname}_{n}")
        cols.append(_as_column(res[0]))
    return TableObject(names, cols)


def _rows2vars(interp, args, nargout):
    t = args[0]
    arr = t.to_array()
    names = [f"Var{k + 1}" for k in range(t.nrows)]
    cols = [Cell.from_list([char(n) for n in t.names], column=True)] + \
        [arr[k:k + 1, :].T.copy() for k in range(t.nrows)]
    return TableObject(["OriginalVariableNames"] + names, cols)


_TABLE_FUNCS = {
    "height": _height_fn, "width": _width_fn, "head": _head, "tail": _tail, "sortrows": _sortrows,
    "removevars": _removevars, "addvars": _addvars, "renamevars": _renamevars,
    "table2array": _table2array, "table2struct": _table2struct, "istable": _istable,
    "summary": _summary, "varfun": _varfun, "rows2vars": _rows2vars,
}

for _name in ("height", "width", "table2array", "table2struct", "removevars", "addvars", "renamevars",
              "varfun", "rows2vars"):
    builtin(_name)(lambda interp, args, nargout, _f=_TABLE_FUNCS[_name]: _f(interp, args, nargout))
builtin("istable")(_istable)


@builtin("head")
def _head_any(interp, args, nargout):
    if isinstance(args[0], TableObject):
        return _head(interp, args, nargout)
    a = args[0]
    k = int(num(args[1]).flat[0]) if len(args) > 1 else 8
    return a[:k, ...]


@builtin("tail")
def _tail_any(interp, args, nargout):
    if isinstance(args[0], TableObject):
        return _tail(interp, args, nargout)
    a = args[0]
    k = int(num(args[1]).flat[0]) if len(args) > 1 else 8
    return a[max(a.shape[0] - k, 0):, ...]


# -- CSV input/output -------------------------------------------------------------------------

def _parse_cell(s: str):
    s = s.strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        low = s.lower()
        if low in ("nan", "inf", "-inf"):
            return float(low)
        return None


@builtin("readtable")
def _readtable(interp, args, nargout):
    fname = text(args[0])
    _, opts = _split_opts(list(args[1:]), {"Delimiter", "ReadVariableNames", "TextType"})
    delim = text(opts["Delimiter"]) if "Delimiter" in opts else None
    with open(fname, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        if delim is None:
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delim = ","
        rows = [r for r in csv.reader(fh, delimiter=delim)]
    rows = [r for r in rows if any(c.strip() for c in r)]
    read_names = not ("ReadVariableNames" in opts and not bool(num(opts["ReadVariableNames"]).flat[0]))
    if read_names and rows:
        header, body = rows[0], rows[1:]
    else:
        header, body = [f"Var{k + 1}" for k in range(len(rows[0]) if rows else 0)], rows
    import re
    names = []
    for k, h in enumerate(header):
        clean = re.sub(r"\W", "_", h.strip()) or f"Var{k + 1}"
        if clean[0].isdigit():
            clean = "x" + clean
        while clean in names:
            clean += "_1"
        names.append(clean)
    as_string = "TextType" in opts and text(opts["TextType"]) == "string"
    cols = []
    for k in range(len(names)):
        raw = [r[k] if k < len(r) else "" for r in body]
        parsed = [_parse_cell(x) for x in raw]
        if all(p is not None for p in parsed):
            cols.append(np.array(parsed, dtype=float).reshape(-1, 1))
        elif as_string:
            d = np.empty((len(raw), 1), dtype=object)
            for i, x in enumerate(raw):
                d[i, 0] = x
            cols.append(StringArray(d))
        else:
            d = np.empty((len(raw), 1), dtype=object)
            for i, x in enumerate(raw):
                d[i, 0] = char(x)
            cols.append(Cell(d))
    return TableObject(names, cols)


@builtin("writetable")
def _writetable(interp, args, nargout):
    t = args[0]
    fname = text(args[1]) if len(args) > 1 else "table.csv"
    _, opts = _split_opts(list(args[2:]), {"Delimiter", "WriteVariableNames"})
    delim = text(opts["Delimiter"]) if "Delimiter" in opts else ","
    delim = {"tab": "\t", "comma": ",", "semi": ";", "space": " "}.get(delim, delim)
    header = []
    for n, c in zip(t.names, t.columns):
        width = c.shape[1] if isinstance(c, np.ndarray) and not is_char(c) else 1
        header += [n] if width == 1 else [f"{n}_{k + 1}" for k in range(width)]
    with open(fname, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delim)
        if not ("WriteVariableNames" in opts and not bool(num(opts["WriteVariableNames"]).flat[0])):
            w.writerow(header)
        for i in range(t.nrows):
            row = []
            for c in t.columns:
                if isinstance(c, Cell):
                    x = c.data[i, 0]
                    row.append(to_str(x) if (is_char(x) or isinstance(x, MString)) else
                               " ".join(f"{v:.15g}" for v in np.asarray(x, dtype=float).flatten()))
                elif isinstance(c, StringArray):
                    row.append(c.data[i, 0] or "")
                elif is_char(c):
                    row.append("".join(c[i]))
                else:
                    row += [("true" if v else "false") if c.dtype == bool else f"{v:.15g}" for v in c[i]]
            w.writerow(row)
    return None
