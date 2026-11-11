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
    Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, string_data, string_result, to_str,
)
from ._registry import builtin
from ._util import bool_scalar, fnum, is_text, num, text


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
    def __init__(self, names: list[str], columns: list[Any], description: str = "",
                 row_names: list[str] | None = None):
        if len(set(names)) != len(names):
            raise MatlabError("Duplicate table variable names.")
        heights = {_height(c) for c in columns}
        if len(heights) > 1:
            raise MatlabError("All table variables must have the same number of rows.")
        self.names = list(names)
        self.columns = list(columns)
        self.description = description
        self.row_names: list[str] | None = None
        if row_names is not None and len(row_names):
            self.set_row_names(list(row_names))

    def set_row_names(self, names: list[str] | None) -> None:
        if not names:
            self.row_names = None
            return
        if self.columns and len(names) != self.nrows:
            raise MatlabError(f"The number of row names ({len(names)}) must match the number of "
                              f"rows in the table ({self.nrows}).")
        if len(set(names)) != len(names):
            raise MatlabError("Duplicate table row names.")
        if any(n == "" for n in names):
            raise MatlabError("Table row names must be nonempty.")
        self.row_names = list(names)

    @property
    def nrows(self) -> int:
        return _height(self.columns[0]) if self.columns else 0

    def class_name(self) -> str:
        return "table"

    def matlab_shape(self) -> tuple:
        return (self.nrows, len(self.names))

    def copy(self) -> "TableObject":
        return TableObject(list(self.names), list(self.columns), self.description, self.row_names)

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
        if is_text(idx) or isinstance(idx, (Cell, StringArray)):
            if self.row_names is None:
                raise MatlabError("This table has no row names; index rows by number or logical mask.")
            out = []
            for n in _names_from(idx):
                if n not in self.row_names:
                    raise MatlabError(f"Unrecognized row name '{n}'.")
                out.append(self.row_names.index(n))
            return np.array(out, dtype=int)
        pos, _ = ix._to_positions(idx, self.nrows, for_set=for_set)
        return pos

    def _sub_row_names(self, rows: np.ndarray) -> list[str] | None:
        if self.row_names is None:
            return None
        return [self.row_names[int(r)] for r in np.asarray(rows).flatten()]

    # -- protocol ----------------------------------------------------------------------------
    _METHODS = {"height", "width", "head", "tail", "summary", "sortrows", "removevars", "addvars",
                "renamevars", "table2array", "table2struct", "istable", "varfun", "rows2vars",
                "join", "innerjoin", "outerjoin", "table2cell", "movevars"}

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
                "RowNames": Cell.from_list([char(n) for n in self.row_names], column=True)
                if self.row_names else Cell(np.empty((0, 1), dtype=object)),
                "DimensionNames": Cell.from_list([char("Row"), char("Variables")]),
            })
        if name == "Row" and self.row_names is not None:
            return Cell.from_list([char(n) for n in self.row_names], column=True)
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
            if "RowNames" in value.fields:
                rn = value.get_field(0, "RowNames")
                empty = (isinstance(rn, np.ndarray) and rn.size == 0) or (isinstance(rn, Cell) and rn.data.size == 0)
                t.set_row_names(None if empty else _names_from(rn))
            return t
        if name == "Row":
            t.set_row_names(None if isinstance(value, np.ndarray) and value.size == 0 else _names_from(value))
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
                           self.description, self._sub_row_names(rows))

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
            return TableObject([self.names[k] for k in keep], [self.columns[k] for k in keep], self.description,
                               self.row_names)
        if args[1] is not ix.COLON:
            raise MatlabError("Delete whole rows with T(rows, :) = [].")
        keep = np.ones(self.nrows, dtype=bool)
        keep[self.row_positions(args[0])] = False
        idx = np.nonzero(keep)[0]
        return TableObject(list(self.names), [_rows(c, idx) for c in self.columns], self.description,
                           self._sub_row_names(idx))

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
        t._pad_row_names(n)
        return t

    def _pad_row_names(self, n: int) -> None:
        if self.row_names is not None and len(self.row_names) < n:
            k = len(self.row_names)
            self.row_names = self.row_names + [f"Row{j + 1}" for j in range(k, n)]

    def brace_set(self, interp, args: list, value: Any) -> Any:
        if len(args) != 2:
            raise MatlabError("Tables are indexed with two subscripts: T{rows, vars} = value.")
        rows = self.row_positions(args[0], for_set=True) if args[0] is not ix.COLON else np.arange(self.nrows)
        cols = self.var_positions(args[1])
        t = self.copy()
        if len(cols) == 1:
            c = cols[0]
            col = t.columns[c]
            if isinstance(col, Cell):
                if len(rows) == 1:
                    piece = Cell(_obj(value))
                elif isinstance(value, Cell) and value.data.size == len(rows):
                    piece = Cell(value.data.reshape(-1, 1))
                else:
                    raise MatlabError("Assign one value per row into a cell table variable.")
            elif isinstance(col, StringArray):
                d = string_data(value)
                if d.size not in (1, len(rows)):
                    raise MatlabError("The number of strings must match the number of rows.")
                piece = StringArray(np.broadcast_to(d.reshape(-1, 1), (len(rows), 1)).copy())
            else:
                vals = value if isinstance(value, np.ndarray) and value.dtype.kind in "biufc" else num(value)
                width = col.shape[1] if col.ndim > 1 else 1
                if vals.size == 1:
                    vals = np.full((len(rows), width), vals.flat[0])
                elif vals.size == len(rows) * width:
                    vals = vals.reshape(len(rows), width)
                else:
                    raise MatlabError("The size of the right-hand side does not match the table subscript.")
                if col.dtype == bool and vals.dtype != bool:
                    col = col.astype(float)
                if vals.dtype.kind == "c" and col.dtype.kind != "c":
                    col = col.astype(complex)
                t.columns[c] = col
                piece = vals
            t.columns[c] = _assign_rows(t.columns[c], rows, piece)
        else:
            if any(not isinstance(t.columns[c], np.ndarray) or is_char(t.columns[c]) for c in cols):
                raise MatlabError("Brace assignment to several variables needs numeric variables.")
            vals = num(value)
            widths = [t.columns[c].shape[1] for c in cols]
            if vals.size == 1:
                vals = np.full((len(rows), sum(widths)), vals.flat[0])
            if vals.shape != (len(rows), sum(widths)):
                raise MatlabError("The size of the right-hand side does not match the table subscript.")
            start = 0
            for c, w in zip(cols, widths):
                col = t.columns[c].astype(float) if t.columns[c].dtype == bool else t.columns[c]
                t.columns[c] = _assign_rows(col, rows, vals[:, start:start + w])
                start += w
        n = max(_height(c) for c in t.columns)
        t.columns = [_pad_rows(c, n) for c in t.columns]
        t._pad_row_names(n)
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
            return TableObject(names, cols, tables[0].description, tables[0].row_names)
        first = tables[0]
        for t in tables[1:]:
            if set(t.names) != set(first.names):
                raise MatlabError("Vertically concatenated tables must have the same variable names.")
        cols = [cat([t.columns[t.names.index(n)] for t in tables], 0) for n in first.names]
        row_names = None
        if all(t.row_names is not None for t in tables):
            row_names = [r for t in tables for r in t.row_names]
        return TableObject(list(first.names), cols, first.description, row_names)

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
        rn_w = max(len(r) for r in self.row_names) if self.row_names else 0

        def pre(label: str) -> str:
            return "    " + (label.ljust(rn_w) + "    " if rn_w else "")

        lines = [f"  {size_str(self.matlab_shape())} table", ""]
        lines.append((pre("") + "    ".join(self.names[j].center(widths[j])
                                           for j in range(len(widths)))).rstrip())
        lines.append(pre("") + "    ".join("_" * widths[j] for j in range(len(widths))))
        lines.append("")
        for k, r in enumerate(cells):
            if n > limit and k == 10:
                lines.append("    " + ":".center(sum(widths) + 4 * (len(widths) - 1) + (rn_w + 4 if rn_w else 0)))
            label = self.row_names[shown[k]] if rn_w else ""
            lines.append((pre(label) + "    ".join(r[j].center(widths[j]) if not r[j][:1].isdigit() and
                                                   r[j][:1] != "-" else r[j].rjust(widths[j])
                                                   for j in range(len(widths)))).rstrip())
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


def _row_names_opt(opts: dict) -> list[str] | None:
    if "RowNames" not in opts:
        return None
    v = opts["RowNames"]
    if isinstance(v, np.ndarray) and v.size == 0:
        return None
    return _names_from(v)


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
        return TableObject(names, cols, row_names=_row_names_opt(opts))
    cols = [_as_column(a) for a in rest]
    arg_names = interp.state.get("call_arg_names") or []
    if "VariableNames" in opts:
        names = _names_from(opts["VariableNames"])
    else:
        names = [arg_names[k] if k < len(arg_names) and arg_names[k] else f"Var{k + 1}" for k in range(len(cols))]
    if len(names) != len(cols):
        raise MatlabError("The number of variable names must match the number of variables.")
    return TableObject(names, cols, row_names=_row_names_opt(opts))


builtin("table")(_table)


@builtin("array2table")
def _array2table(interp, args, nargout):
    rest, opts = _split_opts(args, {"VariableNames", "RowNames"})
    a = rest[0]
    n = a.shape[1]
    names = _names_from(opts["VariableNames"]) if "VariableNames" in opts else [f"Var{k + 1}" for k in range(n)]
    return TableObject(names, [a[:, k:k + 1].copy() for k in range(n)], row_names=_row_names_opt(opts))


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
    return TableObject(names, cols, row_names=_row_names_opt(opts))


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
    directions: list[bool] = []  # True = descending, one per key (or one for all)
    rest = list(args[1:])
    if rest and not (is_text(rest[0]) and text(rest[0]).lower() in ("ascend", "descend")):
        spec = rest.pop(0)
        if is_text(spec) or isinstance(spec, (Cell, StringArray)):
            keys = _names_from(spec)
        else:
            nums = [int(k) for k in num(spec).flatten()]
            keys = [t.names[abs(k) - 1] for k in nums]
            directions = [k < 0 for k in nums]
    if rest:
        d = rest.pop(0)
        words = _names_from(d) if isinstance(d, (Cell, StringArray)) else [text(d)]
        for w in words:
            if w.lower() not in ("ascend", "descend"):
                raise MatlabError(f"sortrows: unknown sort direction '{w}'.")
        directions = [w.lower() == "descend" for w in words]
    if not directions:
        directions = [False] * len(keys)
    elif len(directions) == 1:
        directions = directions * len(keys)
    elif len(directions) != len(keys):
        raise MatlabError("sortrows: give one sort direction, or one per sorting variable.")
    order = np.arange(t.nrows)
    for key, descend in reversed(list(zip(keys, directions))):
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


def _rows2vars(interp, args, nargout):
    t = args[0]
    arr = t.to_array()
    names = [f"Var{k + 1}" for k in range(t.nrows)]
    cols = [Cell.from_list([char(n) for n in t.names], column=True)] + \
        [arr[k:k + 1, :].T.copy() for k in range(t.nrows)]
    return TableObject(["OriginalVariableNames"] + names, cols)


# -- grouping, joins and missing data ------------------------------------------------------------

_MISSING = object()  # key used for a missing value (NaN, '' in a cellstr, <missing> string)


def _is_text_value(x: Any) -> bool:
    return isinstance(x, (str, MString)) or is_char(x)


def _col_keys(col: Any) -> list:
    """One hashable, sortable key per row of a table variable or grouping vector."""
    if isinstance(col, Cell):
        keys = []
        for i in range(col.data.shape[0]):
            x = col.data[i, 0] if col.data.ndim == 2 else col.data[i]
            if _is_text_value(x):
                s = to_str(x)
                keys.append(_MISSING if s == "" else s)
            elif isinstance(x, np.ndarray) and x.size == 1 and x.dtype.kind in "biuf":
                v = float(x.flat[0])
                keys.append(_MISSING if np.isnan(v) else v)
            elif isinstance(x, np.ndarray) and x.size == 0:
                keys.append(_MISSING)
            else:
                raise MatlabError("Grouping cell arrays must hold character vectors or numeric scalars.")
        return keys
    if isinstance(col, StringArray):
        return [_MISSING if x is None else x for x in col.data[:, 0]]
    if isinstance(col, MString):
        return [str(col)]
    if is_char(col):
        return ["".join(r).rstrip() for r in col]
    a = num(col)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    keys = []
    for i in range(a.shape[0]):
        row = a[i].flatten()
        if a.dtype.kind in "fc" and np.any(np.isnan(row)):
            keys.append(_MISSING)
        elif row.size == 1:
            keys.append(row[0].item() if hasattr(row[0], "item") else row[0])
        else:
            keys.append(tuple(row.tolist()))
    return keys


def _sort_key(k: Any):
    if isinstance(k, str):
        return (1, k)
    return (0, k)


def _group_rows(cols: list) -> tuple[np.ndarray, list[int]]:
    """Group numbers (1-based float, NaN for rows with a missing key) and the first row of each
    group, with the groups in sorted order of their key values."""
    per_col = [_col_keys(c) for c in cols]
    n = len(per_col[0]) if per_col else 0
    if any(len(k) != n for k in per_col):
        raise MatlabError("Grouping variables must have the same number of rows.")
    first: dict[tuple, int] = {}
    rows_keys = []
    for i in range(n):
        key = tuple(k[i] for k in per_col)
        rows_keys.append(key)
        if any(x is _MISSING for x in key):
            continue
        first.setdefault(key, i)
    ordered = sorted(first, key=lambda key: tuple(_sort_key(x) for x in key))
    number = {key: g + 1.0 for g, key in enumerate(ordered)}
    gidx = np.array([number.get(key, np.nan) for key in rows_keys], dtype=float)
    return gidx, [first[key] for key in ordered]


def _orient(v: Any, row: bool) -> Any:
    if not row:
        return v
    if isinstance(v, Cell):
        return Cell(v.data.reshape(1, -1))
    if isinstance(v, StringArray):
        return StringArray(v.data.reshape(1, -1))
    if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
        return v.reshape(1, -1)
    return v


def _is_row_vector(v: Any) -> bool:
    shape = v.data.shape if isinstance(v, (Cell, StringArray)) else (np.shape(v) if isinstance(v, np.ndarray)
                                                                     else (1, 1))
    return len(shape) == 2 and shape[0] == 1 and shape[1] > 1 and not is_char(v)


@builtin("findgroups")
def _findgroups(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for findgroups.")
    if len(args) == 1 and isinstance(args[0], TableObject):
        t = args[0]
        g, first = _group_rows(t.columns)
        G = g.reshape(-1, 1)
        if nargout > 1:
            pos = np.array(first, dtype=int)
            return G, TableObject(list(t.names), [_rows(c, pos) for c in t.columns])
        return G
    row = all(_is_row_vector(a) for a in args)
    cols = [_as_column(a) for a in args]
    g, first = _group_rows(cols)
    G = _orient(g.reshape(-1, 1), row)
    if nargout <= 1:
        return G
    pos = np.array(first, dtype=int)
    ids = [_orient(_rows(c, pos), row) for c in cols]
    return (G, *ids[:nargout - 1])


def _take_rows(v: Any, pos: np.ndarray, as_vector: bool) -> Any:
    if isinstance(v, TableObject):
        return v.index_get(None, [pos.reshape(1, -1) + 1.0, ix.COLON])
    if as_vector:
        if isinstance(v, (Cell, StringArray)):
            flat = v.data.flatten(order="F")[pos]
            data = flat.reshape(1, -1) if v.data.shape[0] == 1 else flat.reshape(-1, 1)
            return type(v)(data)
        flat = v.flatten(order="F")[pos]
        return flat.reshape(1, -1) if v.shape[0] == 1 else flat.reshape(-1, 1)
    return _rows(v, pos)


def _stack_results(results: list) -> Any:
    """Concatenate per-group results vertically (scalars give a column)."""
    if not results:
        return np.zeros((0, 1))
    if all(isinstance(r, Cell) for r in results):
        return Cell(np.vstack([r.data.reshape(1, -1) if r.data.ndim == 2 and r.data.shape[0] == 1 else r.data
                               for r in results]))
    if all(isinstance(r, (MString, StringArray)) for r in results):
        return StringArray(np.vstack([string_data(r).reshape(1, -1) for r in results]))
    if all(isinstance(r, np.ndarray) and not is_char(r) for r in results):
        rows = [r.reshape(1, -1) if r.ndim == 2 and r.shape[0] == 1 or r.size == 1 else None for r in results]
        if any(x is None for x in rows) or len({x.shape[1] for x in rows}) > 1:
            raise MatlabError("The function must return results of the same size for every group; "
                              "return a cell, e.g. @(x) {x}, for non-uniform results.")
        out = np.vstack(rows)
        return out
    raise MatlabError("The function must return numeric, logical, string or cell results; "
                      "return a cell, e.g. @(x) {x}, for other results.")


@builtin("splitapply")
def _splitapply(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("splitapply expects (fun, X, G).")
    fun = args[0]
    graw = args[-1]
    G = num(graw).flatten(order="F")
    data = []
    for a in args[1:-1]:
        if isinstance(a, TableObject):
            data.extend(a.columns)
        else:
            data.append(a)
    if G.size == 0:
        return np.zeros((0, 1))
    valid = G[~np.isnan(G)]
    ngroups = int(valid.max()) if valid.size else 0
    if valid.size and (np.any(valid < 1) or np.any(valid != np.round(valid))):
        raise MatlabError("Group numbers must be positive integers, as returned by findgroups.")
    present = set(int(x) for x in valid)
    if present != set(range(1, ngroups + 1)):
        raise MatlabError("Group numbers must run from 1 to the number of groups with none skipped.")
    vec_mode = []
    for d in data:
        shape = d.data.shape if isinstance(d, (Cell, StringArray)) else np.shape(d)
        is_vec = len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)
        if is_vec and int(np.prod(shape)) == G.size:
            vec_mode.append(True)
        elif _height(d) == G.size:
            vec_mode.append(False)
        else:
            raise MatlabError("Each data argument must have one row (or element) per group number.")
    nout = max(nargout, 1)
    collected: list[list] = [[] for _ in range(nout)]
    for g in range(1, ngroups + 1):
        pos = np.nonzero(G == g)[0]
        pieces = [_take_rows(d, pos, vm) for d, vm in zip(data, vec_mode)]
        res = interp.call_function_value(fun, pieces, nout)
        if len(res) < nout:
            raise MatlabError("The function returned fewer outputs than requested.")
        for k in range(nout):
            collected[k].append(res[k])
    row = isinstance(graw, np.ndarray) and graw.ndim == 2 and graw.shape[0] == 1 and graw.shape[1] > 1
    outs = []
    for k in range(nout):
        y = _stack_results(collected[k])
        if row and isinstance(y, np.ndarray) and y.shape[1] == 1:
            y = y.reshape(1, -1)
        outs.append(y)
    return tuple(outs) if nout > 1 else outs[0]


def _nan_reduce(fn, empty=np.nan):
    def f(a: np.ndarray) -> np.ndarray:
        a = a.astype(float) if a.dtype.kind in "biu" else a
        out = []
        for j in range(a.shape[1]):
            col = a[:, j]
            col = col[~np.isnan(col)] if col.dtype.kind in "fc" else col
            out.append(fn(col) if col.size else empty)
        return np.array(out, dtype=float).reshape(1, -1)
    return f


def _mode_of(col: np.ndarray) -> float:
    vals, counts = np.unique(col, return_counts=True)
    return float(vals[np.argmax(counts)])


_GROUP_METHODS = {
    "mean": _nan_reduce(np.mean),
    "sum": _nan_reduce(np.sum, 0.0),
    "min": _nan_reduce(np.min),
    "max": _nan_reduce(np.max),
    "median": _nan_reduce(np.median),
    "mode": _nan_reduce(_mode_of),
    "std": _nan_reduce(lambda c: np.std(c, ddof=1) if c.size > 1 else 0.0),
    "var": _nan_reduce(lambda c: np.var(c, ddof=1) if c.size > 1 else 0.0),
    "range": _nan_reduce(lambda c: np.max(c) - np.min(c)),
    "nnz": lambda a: np.count_nonzero((a != 0) & ~np.isnan(a.astype(float)), axis=0).reshape(1, -1).astype(float),
    "numel": lambda a: np.full((1, a.shape[1]), float(a.shape[0])),
    "nummissing": lambda a: np.sum(np.isnan(a.astype(float)), axis=0).reshape(1, -1).astype(float),
}


def _method_list(v: Any) -> list:
    if isinstance(v, Cell):
        return [x if isinstance(x, FunctionHandle) else to_str(x).lower() for x in v.values_f()]
    if isinstance(v, StringArray):
        return [x.lower() for x in v.values_f()]
    if isinstance(v, FunctionHandle):
        return [v]
    m = text(v).lower()
    if m == "all":
        return ["mean", "sum", "min", "max", "range", "median", "mode", "var", "std", "nummissing", "nnz"]
    return [m]


def _apply_method(interp, method, piece: Any) -> Any:
    if isinstance(method, FunctionHandle):
        res = interp.call_function_value(method, [piece], 1)
        return res[0]
    fn = _GROUP_METHODS.get(method)
    if fn is None:
        raise MatlabError(f"Unknown summary method '{method}'.")
    if not isinstance(piece, np.ndarray) or is_char(piece):
        if method == "numel":
            return float(_height(piece))
        raise MatlabError(f"The method '{method}' needs numeric or logical data.")
    a = piece.reshape(-1, 1) if piece.ndim == 1 else piece
    return fn(a)


def _method_name(method, k: int) -> str:
    if isinstance(method, FunctionHandle):
        return f"fun{k + 1}"
    return method


def _is_numeric_var(c: Any) -> bool:
    return isinstance(c, np.ndarray) and c.dtype.kind in "biufc"


def _summarize(interp, t: "TableObject", group_names: list[str], methods: list, data_names: list[str],
               count: bool = True) -> "TableObject":
    gcols = [t.columns[t.var_index(n)] for n in group_names]
    gidx, first = _group_rows(gcols)
    pos = np.array(first, dtype=int)
    names = list(group_names)
    cols = [_rows(c, pos) for c in gcols]
    members = [np.nonzero(gidx == g + 1)[0] for g in range(len(first))]
    if count:
        names.append("GroupCount")
        cols.append(np.array([[float(m.size)] for m in members]).reshape(-1, 1))
    for dn in data_names:
        col = t.columns[t.var_index(dn)]
        for k, method in enumerate(methods):
            results = [_apply_method(interp, method, _rows(col, m)) for m in members]
            results = [r if not isinstance(r, (int, float)) else np.array([[float(r)]]) for r in results]
            names.append(f"{_method_name(method, k)}_{dn}")
            cols.append(_stack_results(results) if results else np.zeros((0, 1)))
    return TableObject(names, cols)


@builtin("groupsummary")
def _groupsummary(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("groupsummary expects (T, groupvars, ...).")
    if not isinstance(args[0], TableObject):
        # array form: groupsummary(X, G, method)
        X = args[0]
        Gv = args[1]
        methods = _method_list(args[2]) if len(args) > 2 else []
        Xc = _as_column(X)
        t = TableObject(["G", "X"], [_as_column(Gv), Xc])
        res = _summarize(interp, t, ["G"], methods, ["X"] if methods else [])
        B = np.hstack([fnum(c) for c in res.columns[2:]]) if methods else np.zeros((len(res.columns[1]), 0))
        return (B, res.columns[0], res.columns[1])[:max(nargout, 1)] if nargout > 1 else B
    t = args[0]
    rest, opts = _split_opts(list(args[1:]), {"IncludeMissingGroups", "IncludeEmptyGroups"})
    group_names = [t.names[p] for p in t.var_positions(rest[0])]
    methods = _method_list(rest[1]) if len(rest) > 1 else []
    if len(rest) > 2:
        data_names = [t.names[p] for p in t.var_positions(rest[2])]
    else:
        data_names = [n for n, c in zip(t.names, t.columns) if n not in group_names and _is_numeric_var(c)] \
            if methods else []
    return _summarize(interp, t, group_names, methods, data_names)


def _varfun(interp, args, nargout):
    fn, t = args[0], args[1]
    if not isinstance(t, TableObject):
        raise MatlabError("varfun expects a function and a table.")
    _, opts = _split_opts(list(args[2:]), {"GroupingVariables", "InputVariables", "OutputFormat"})
    fname = fn.name if getattr(fn, "name", None) else "Fun"
    groups = [t.names[p] for p in t.var_positions(opts["GroupingVariables"])] \
        if "GroupingVariables" in opts else []
    if "InputVariables" in opts:
        inputs = [t.names[p] for p in t.var_positions(opts["InputVariables"])]
    else:
        inputs = [n for n in t.names if n not in groups]
    fmt = text(opts["OutputFormat"]).lower() if "OutputFormat" in opts else "table"
    if groups:
        res = _summarize(interp, t, groups, [fn], inputs)
        res.names = res.names[:len(groups) + 1] + [f"{fname}_{n}" for n in inputs]
        if fmt == "uniform":
            return np.hstack([fnum(c) for c in res.columns[len(groups) + 1:]])
        return res
    results = []
    for n in inputs:
        r = interp.call_function_value(fn, [t.columns[t.var_index(n)]], 1)
        if not r:
            raise MatlabError("The function must return a value.")
        results.append(r[0])
    if fmt == "uniform":
        if not all(isinstance(r, np.ndarray) and r.size == 1 for r in results):
            raise MatlabError("With 'OutputFormat' 'uniform' the function must return scalars.")
        return np.array([[float(np.real(r.flat[0])) for r in results]]).reshape(1, -1)
    if fmt == "cell":
        return Cell.from_list(results)
    return TableObject([f"{fname}_{n}" for n in inputs], [_as_column(r) for r in results])


def _missing_like(col: Any, n: int) -> Any:
    if isinstance(col, Cell):
        d = np.empty((n, 1), dtype=object)
        for i in range(n):
            d[i, 0] = char("")
        return Cell(d)
    if isinstance(col, StringArray):
        return StringArray(np.empty((n, 1), dtype=object))
    if is_char(col):
        return np.full((n, col.shape[1]), " ", dtype="<U1")
    width = col.shape[1] if col.ndim > 1 else 1
    return np.full((n, width), np.nan, dtype=complex if col.dtype.kind == "c" else float)


def _take_with_missing(col: Any, idx: list[int]) -> Any:
    """Rows of a variable, with -1 standing for a missing row."""
    idx_arr = np.array(idx, dtype=int)
    present = idx_arr >= 0
    if present.all():
        return _rows(col, idx_arr)
    if isinstance(col, np.ndarray) and not is_char(col) and col.dtype.kind in "biu":
        col = col.astype(float)
    out = _missing_like(col, len(idx))
    if present.any():
        got = _rows(col, idx_arr[present])
        where = np.nonzero(present)[0]
        if isinstance(out, (Cell, StringArray)):
            out.data[where, ...] = got.data
        else:
            out[where, ...] = got
    return out


def _join_keys(t1: "TableObject", t2: "TableObject", opts: dict, default_common: bool):
    if "Keys" in opts:
        k1 = [t1.names[p] for p in t1.var_positions(opts["Keys"])]
        k2 = [t2.names[p] for p in t2.var_positions(opts["Keys"])]
    elif "LeftKeys" in opts or "RightKeys" in opts:
        if not ("LeftKeys" in opts and "RightKeys" in opts):
            raise MatlabError("Specify both 'LeftKeys' and 'RightKeys'.")
        k1 = [t1.names[p] for p in t1.var_positions(opts["LeftKeys"])]
        k2 = [t2.names[p] for p in t2.var_positions(opts["RightKeys"])]
    else:
        k1 = [n for n in t1.names if n in t2.names]
        k2 = list(k1)
        if not k1:
            raise MatlabError("The tables have no variable names in common to use as keys; "
                              "specify them with 'Keys'.")
    if len(k1) != len(k2):
        raise MatlabError("The left and right keys must have the same number of variables.")
    return k1, k2


def _row_keys(t: "TableObject", keys: list[str]) -> list[tuple]:
    per = [_col_keys(t.columns[t.var_index(k)]) for k in keys]
    return [tuple(p[i] for p in per) for i in range(t.nrows)]


def _joined_table(t1, t2, left_idx, right_idx, left_vars, right_vars, merged_keys=None):
    names, cols = [], []
    right_set = set(right_vars)
    left_set = set(left_vars)
    for n in left_vars:
        col = t1.columns[t1.var_index(n)]
        if merged_keys and n in merged_keys:
            rcol = t2.columns[t2.var_index(merged_keys[n])]
            lpart = _take_with_missing(col, left_idx)
            rpart = _take_with_missing(rcol, right_idx)
            lmiss = np.array(left_idx) < 0
            if isinstance(lpart, (Cell, StringArray)):
                lpart.data[lmiss, ...] = rpart.data[lmiss, ...]
            else:
                lpart = lpart.astype(float) if lpart.dtype.kind in "biu" else lpart
                lpart[lmiss, ...] = rpart[lmiss, ...]
            names.append(n)
            cols.append(lpart)
            continue
        names.append(n + "_left" if n in right_set else n)
        cols.append(_take_with_missing(col, left_idx))
    for n in right_vars:
        names.append(n + "_right" if n in left_set else n)
        cols.append(_take_with_missing(t2.columns[t2.var_index(n)], right_idx))
    return TableObject(names, cols)


_JOIN_OPTS = {"Keys", "LeftKeys", "RightKeys", "LeftVariables", "RightVariables", "MergeKeys", "Type"}


def _check_tables(args: list, name: str) -> tuple:
    if len(args) < 2 or not isinstance(args[0], TableObject) or not isinstance(args[1], TableObject):
        raise MatlabError(f"{name} expects two tables.")
    _, opts = _split_opts(list(args[2:]), _JOIN_OPTS)
    return args[0], args[1], opts


def _side_vars(t, opts, key: str, default: list[str]) -> list[str]:
    if key in opts:
        return [t.names[p] for p in t.var_positions(opts[key])]
    return default


def _innerjoin(interp, args, nargout):
    t1, t2, opts = _check_tables(args, "innerjoin")
    k1, k2 = _join_keys(t1, t2, opts, True)
    rk1, rk2 = _row_keys(t1, k1), _row_keys(t2, k2)
    where: dict[tuple, list[int]] = {}
    for j, key in enumerate(rk2):
        if not any(x is _MISSING for x in key):
            where.setdefault(key, []).append(j)
    pairs = [(i, j) for i, key in enumerate(rk1) for j in where.get(key, [])]
    pairs.sort(key=lambda p: (tuple(_sort_key(x) for x in rk1[p[0]]), p[0], p[1]))
    li = [p[0] for p in pairs]
    ri = [p[1] for p in pairs]
    left_vars = _side_vars(t1, opts, "LeftVariables", list(t1.names))
    right_vars = _side_vars(t2, opts, "RightVariables", [n for n in t2.names if n not in k2])
    res = _joined_table(t1, t2, li, ri, left_vars, right_vars)
    if nargout > 1:
        return res, (np.array(li, dtype=float) + 1).reshape(-1, 1), (np.array(ri, dtype=float) + 1).reshape(-1, 1)
    return res


def _join(interp, args, nargout):
    t1, t2, opts = _check_tables(args, "join")
    k1, k2 = _join_keys(t1, t2, opts, True)
    rk1, rk2 = _row_keys(t1, k1), _row_keys(t2, k2)
    where: dict[tuple, int] = {}
    for j, key in enumerate(rk2):
        if key in where:
            raise MatlabError("The key values of the right table must be unique for join; use innerjoin "
                              "or outerjoin for one-to-many matches.")
        where[key] = j
    ri = []
    for key in rk1:
        if key not in where:
            raise MatlabError("Every key value of the left table must appear in the right table; "
                              "use outerjoin to keep unmatched rows.")
        ri.append(where[key])
    left_vars = _side_vars(t1, opts, "LeftVariables", list(t1.names))
    right_vars = _side_vars(t2, opts, "RightVariables", [n for n in t2.names if n not in k2])
    res = _joined_table(t1, t2, list(range(t1.nrows)), ri, left_vars, right_vars)
    res.row_names = list(t1.row_names) if t1.row_names else None
    if nargout > 1:
        return res, (np.array(ri, dtype=float) + 1).reshape(-1, 1)
    return res


def _outerjoin(interp, args, nargout):
    t1, t2, opts = _check_tables(args, "outerjoin")
    k1, k2 = _join_keys(t1, t2, opts, True)
    kind = text(opts["Type"]).lower() if "Type" in opts else "full"
    if kind not in ("full", "left", "right"):
        raise MatlabError("'Type' must be 'full', 'left' or 'right'.")
    merge = "MergeKeys" in opts and bool(num(opts["MergeKeys"]).flat[0])
    rk1, rk2 = _row_keys(t1, k1), _row_keys(t2, k2)
    where: dict[tuple, list[int]] = {}
    for j, key in enumerate(rk2):
        where.setdefault(key, []).append(j)
    pairs = []
    matched_right = set()
    for i, key in enumerate(rk1):
        js = [] if any(x is _MISSING for x in key) else where.get(key, [])
        for j in js:
            pairs.append((i, j, key))
            matched_right.add(j)
        if not js and kind in ("full", "left"):
            pairs.append((i, -1, key))
    if kind in ("full", "right"):
        for j, key in enumerate(rk2):
            if j not in matched_right:
                pairs.append((-1, j, key))
    pairs.sort(key=lambda p: (tuple((2, 0) if x is _MISSING else _sort_key(x) for x in p[2]),
                              p[0] if p[0] >= 0 else 1 << 30, p[1]))
    li = [p[0] for p in pairs]
    ri = [p[1] for p in pairs]
    left_vars = _side_vars(t1, opts, "LeftVariables", list(t1.names))
    if merge:
        right_default = [n for n in t2.names if n not in k2]
        merged = dict(zip(k1, k2))
    else:
        right_default = list(t2.names)
        merged = None
    right_vars = _side_vars(t2, opts, "RightVariables", right_default)
    res = _joined_table(t1, t2, li, ri, left_vars, right_vars, merged)
    if nargout > 1:
        return res, (np.array(li, dtype=float) + 1).reshape(-1, 1), (np.array(ri, dtype=float) + 1).reshape(-1, 1)
    return res


def _table2cell(interp, args, nargout):
    t = args[0]
    data = np.empty((t.nrows, len(t.names)), dtype=object)
    for j, c in enumerate(t.columns):
        for i in range(t.nrows):
            if isinstance(c, Cell):
                data[i, j] = c.data[i, 0]
            elif isinstance(c, StringArray):
                data[i, j] = string_result(c.data[i:i + 1, :])
            elif is_char(c):
                data[i, j] = char("".join(c[i]))
            else:
                data[i, j] = c[i:i + 1, :].copy()
    return Cell(data)


def _movevars(interp, args, nargout):
    t = args[0]
    if len(args) < 2:
        raise MatlabError("movevars expects (T, vars, 'Before'|'After', location).")
    moving = [t.names[p] for p in t.var_positions(args[1])]
    _, opts = _split_opts(list(args[2:]), {"Before", "After"})
    rest = [n for n in t.names if n not in moving]
    if "Before" in opts or "After" in opts:
        where = opts.get("Before", opts.get("After"))
        target = t.names[t.var_positions(where)[0]]
        if target in moving:
            raise MatlabError("The location cannot be one of the variables being moved.")
        k = rest.index(target) + (1 if "After" in opts else 0)
    else:
        k = len(rest)
    order = rest[:k] + moving + rest[k:]
    return TableObject(order, [t.columns[t.var_index(n)] for n in order], t.description, t.row_names)


# -- missing data -------------------------------------------------------------------------------

def _missing_mask(v: Any) -> np.ndarray:
    """Elementwise missing flags (same shape as the value's data)."""
    if isinstance(v, Cell):
        return np.vectorize(lambda x: (_is_text_value(x) and to_str(x) == "") or
                            (isinstance(x, np.ndarray) and x.size == 0), otypes=[bool])(v.data) \
            if v.data.size else np.zeros(v.data.shape, dtype=bool)
    if isinstance(v, StringArray):
        return np.vectorize(lambda x: x is None, otypes=[bool])(v.data) if v.data.size else \
            np.zeros(v.data.shape, dtype=bool)
    if isinstance(v, MString):
        return np.zeros((1, 1), dtype=bool)
    if is_char(v):
        return v == " "
    a = num(v)
    return np.isnan(a) if a.dtype.kind in "fc" else np.zeros(a.shape, dtype=bool)


def _row_missing(v: Any) -> np.ndarray:
    m = _missing_mask(v)
    return m.reshape(m.shape[0], -1).sum(axis=1) if m.ndim > 1 else m.astype(int)


@builtin("rmmissing")
def _rmmissing(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for rmmissing.")
    v = args[0]
    rest, opts = _split_opts(list(args[1:]), {"MinNumMissing", "DataVariables"})
    min_missing = int(num(opts["MinNumMissing"]).flat[0]) if "MinNumMissing" in opts else 1
    if isinstance(v, TableObject):
        names = [v.names[p] for p in v.var_positions(opts["DataVariables"])] if "DataVariables" in opts \
            else v.names
        counts = np.zeros(v.nrows, dtype=int)
        for n in names:
            counts += _row_missing(v.columns[v.var_index(n)])
        drop = counts >= min_missing
        keep = np.nonzero(~drop)[0]
        res = v.index_get(interp, [keep.reshape(1, -1) + 1.0, ix.COLON])
        return (res, drop.reshape(-1, 1)) if nargout > 1 else res
    data = v.data if isinstance(v, (Cell, StringArray)) else v
    if not isinstance(data, np.ndarray):
        return (v, np.zeros((1, 1), dtype=bool)) if nargout > 1 else v
    mask = _missing_mask(v)
    dim = int(num(rest[0]).flat[0]) if rest else None
    is_vec = data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1) and dim is None
    if is_vec:
        drop = mask.flatten(order="F") if min_missing <= 1 else np.zeros(mask.size, dtype=bool)
        keep = ~drop
        flat = data.flatten(order="F")[keep]
        out = flat.reshape(1, -1) if data.shape[0] == 1 else flat.reshape(-1, 1)
        tf = drop.reshape(data.shape, order="F")
    else:
        axis = 0 if dim in (None, 1) else 1
        counts = mask.sum(axis=1 - axis)
        drop = counts >= min_missing
        out = np.take(data, np.nonzero(~drop)[0], axis=axis)
        tf = drop.reshape(-1, 1) if axis == 0 else drop.reshape(1, -1)
    res = type(v)(out) if isinstance(v, (Cell, StringArray)) else out
    return (res, tf) if nargout > 1 else res


def _fill_vector(vals: list, miss: np.ndarray, method: str, const: Any, endv: Any) -> tuple[list, np.ndarray]:
    """Fill the missing entries of one 1-D slice. ``vals`` is a list of Python values."""
    n = len(vals)
    known = np.nonzero(~miss)[0]
    out = list(vals)
    filled = np.zeros(n, dtype=bool)
    if method == "constant":
        for i in np.nonzero(miss)[0]:
            out[i] = const
            filled[i] = True
        return out, filled
    if known.size == 0:
        return out, filled
    first, last = known[0], known[-1]
    for i in np.nonzero(miss)[0]:
        prev = known[known < i]
        nxt = known[known > i]
        p = prev[-1] if prev.size else None
        q = nxt[0] if nxt.size else None
        interior = p is not None and q is not None
        if not interior and endv is not None and method not in ("previous", "next"):
            if isinstance(endv, str):
                if endv == "none":
                    continue
                if endv == "nearest":
                    out[i] = vals[first] if i < first else vals[last]
                    filled[i] = True
                    continue
                if endv == "previous":
                    if p is not None:
                        out[i], filled[i] = vals[p], True
                    continue
                if endv == "next":
                    if q is not None:
                        out[i], filled[i] = vals[q], True
                    continue
            else:
                out[i], filled[i] = endv, True
                continue
        if method == "previous":
            if p is not None:
                out[i], filled[i] = vals[p], True
        elif method == "next":
            if q is not None:
                out[i], filled[i] = vals[q], True
        elif method == "nearest":
            if p is None or (q is not None and (q - i) <= (i - p)):
                out[i] = vals[q]
            else:
                out[i] = vals[p]
            filled[i] = True
    if method in ("linear", "spline", "pchip", "makima"):
        if known.size >= 2:
            from .numerics import _interp1_object
            x = known.astype(float)
            y = np.array([vals[k] for k in known])
            f = _interp1_object(method, x, y)
            for i in np.nonzero(miss)[0]:
                interior = first < i < last
                if not interior and endv is not None:
                    continue  # handled above (or deliberately left missing)
                val = f(np.array([float(i)]))[0]
                out[i] = complex(val) if np.iscomplexobj(y) else float(val)
                filled[i] = True
        elif known.size == 1:
            for i in np.nonzero(miss)[0]:
                if endv is None or (isinstance(endv, str) and endv == "extrap"):
                    out[i], filled[i] = vals[known[0]], True
    return out, filled


def _fill_values(v: Any, method: str, const: Any, endv: Any, axis: int | None) -> tuple[Any, np.ndarray]:
    mask = _missing_mask(v)
    if isinstance(v, (Cell, StringArray)):
        data = v.data
    else:
        data = num(v) if not is_char(v) else v
        if isinstance(data, np.ndarray) and data.dtype.kind in "biu":
            return v, np.zeros(data.shape, dtype=bool)
    if axis is None:
        axis = 1 if data.ndim == 2 and data.shape[0] == 1 else 0
    out = data.copy()
    filled = np.zeros(data.shape, dtype=bool)
    if method == "constant":
        if isinstance(v, Cell):
            const = char(text(const)) if is_text(const) else const
        elif isinstance(v, StringArray):
            const = text(const)
        elif is_char(v):
            const = text(const)
        else:
            c = fnum(const)
            const = c.flat[0] if c.size else np.nan
    moved = np.moveaxis(out, axis, 0)
    mmask = np.moveaxis(mask, axis, 0)
    fmask = np.moveaxis(filled, axis, 0)
    for idx in np.ndindex(moved.shape[1:]):
        sl = (slice(None),) + idx
        vals = list(moved[sl])
        newvals, f = _fill_vector(vals, mmask[sl], method, const, endv)
        if f.any():
            col = moved[sl]
            for k in np.nonzero(f)[0]:
                col[k] = newvals[k]
            fmask[sl] = f
    if isinstance(v, (Cell, StringArray)):
        return type(v)(out), filled
    return out, filled


@builtin("fillmissing")
def _fillmissing(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("fillmissing expects (A, method, ...).")
    v = args[0]
    method = text(args[1]).lower()
    rest = list(args[2:])
    const = None
    if method == "constant":
        if not rest:
            raise MatlabError("fillmissing(A, 'constant', v) needs a fill value.")
        const = rest.pop(0)
    elif method not in ("previous", "next", "nearest", "linear", "spline", "pchip", "makima"):
        raise MatlabError(f"Unknown fill method '{method}'.")
    axis = None
    if rest and not is_text(rest[0]):
        axis = int(num(rest.pop(0)).flat[0]) - 1
    _, opts = _split_opts(rest, {"EndValues", "DataVariables", "SamplePoints", "MissingLocations"})
    endv = None
    if "EndValues" in opts:
        e = opts["EndValues"]
        if is_text(e):
            endv = text(e).lower()
            if endv == "extrap":
                endv = None
        else:
            endv = float(fnum(e).flat[0])
    if isinstance(v, TableObject):
        names = [v.names[p] for p in v.var_positions(opts["DataVariables"])] if "DataVariables" in opts \
            else v.names
        t = v.copy()
        masks = []
        for k, n in enumerate(t.names):
            col = t.columns[k]
            if n not in names:
                masks.append(np.zeros((t.nrows, 1), dtype=bool))
                continue
            c = const
            if method == "constant" and isinstance(const, Cell) and const.data.size == len(names):
                c = const.data.flatten(order="F")[names.index(n)]
            numeric_col = isinstance(col, np.ndarray) and not is_char(col)
            if method in ("linear", "spline", "pchip", "makima") and not numeric_col:
                masks.append(np.zeros((t.nrows, 1), dtype=bool))
                continue
            if method == "constant" and numeric_col and is_text(c):
                masks.append(np.zeros((t.nrows, 1), dtype=bool))
                continue
            t.columns[k], m = _fill_values(col, method, c, endv, 0)
            masks.append(m.reshape(t.nrows, -1).any(axis=1).reshape(-1, 1))
        if nargout > 1:
            return t, np.hstack(masks) if masks else np.zeros((t.nrows, 0), dtype=bool)
        return t
    res, filled = _fill_values(v, method, const, endv, axis)
    return (res, filled) if nargout > 1 else res


_TABLE_FUNCS = {
    "height": _height_fn, "width": _width_fn, "head": _head, "tail": _tail, "sortrows": _sortrows,
    "removevars": _removevars, "addvars": _addvars, "renamevars": _renamevars,
    "table2array": _table2array, "table2struct": _table2struct, "istable": _istable,
    "summary": _summary, "varfun": _varfun, "rows2vars": _rows2vars,
    "join": _join, "innerjoin": _innerjoin, "outerjoin": _outerjoin, "table2cell": _table2cell,
    "movevars": _movevars,
}

for _name in ("height", "width", "table2array", "table2struct", "removevars", "addvars", "renamevars",
              "varfun", "rows2vars", "innerjoin", "outerjoin", "table2cell", "movevars"):
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
