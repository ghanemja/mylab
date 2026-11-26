"""Timetables: tables whose rows are labelled with times (datetime or duration values).

A timetable is a table with a column of row times in front. Rows can be selected by time
(``TT(timerange(a, b), :)``, ``TT(withtol(t, tol), :)``, a datetime or date text), and
``retime`` and ``synchronize`` move the data to other row times by filling, interpolating or
aggregating.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, to_numeric
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, is_text, nargs, text
from .table import _TABLE_FUNCS, TableObject, _as_column, _height, _missing_mask, _names_from, _rows
from .tables_more import _column_from_values, _flag, _options

DAY = 86400.0
_STEPS = {"yearly": "year", "quarterly": "quarter", "monthly": "month", "weekly": "week", "daily": "day",
          "hourly": "hour", "minutely": "minute", "secondly": "second"}
_FILL = ("fillwithmissing", "fillwithconstant", "previous", "next", "nearest", "default")
_INTERP = ("linear", "spline", "pchip", "makima")
_AGG = ("sum", "mean", "prod", "min", "max", "count", "firstvalue", "lastvalue", "median", "mode", "std", "var")


def _is_times(v: Any) -> bool:
    return getattr(v, "kind", None) in ("datetime", "duration")


def _time_column(v: Any, name: str) -> Any:
    if not _is_times(v):
        raise MatlabError(f"{name}: the row times must be a datetime or duration vector.",
                          "MATLAB:timetable:InvalidRowTimes")
    data = v.data
    if data.ndim != 2 or (data.shape[0] != 1 and data.shape[1] != 1 and data.size):
        raise MatlabError(f"{name}: the row times must be a vector.", "MATLAB:timetable:InvalidRowTimes")
    return v.like(data.reshape(-1, 1).copy())


class TimetableObject(TableObject):
    """A MATLAB timetable."""

    DEFAULT_DIMS = ("Time", "Variables")
    kind_word = "timetable"

    def __init__(self, names: list[str], columns: list, row_times: Any, description: str = ""):
        self.row_times = row_times
        super().__init__(names, columns, description, None)
        if columns and _height(columns[0]) != row_times.data.shape[0]:
            raise MatlabError("The number of row times must match the number of rows of the variables.",
                              "MATLAB:timetable:RowTimesLengthMismatch")

    @property
    def nrows(self) -> int:
        return int(self.row_times.data.shape[0])

    def class_name(self) -> str:
        return "timetable"

    def isa_class(self, name: str) -> bool:
        return name in ("timetable", "tabular")

    def copy(self) -> "TimetableObject":
        return TimetableObject(list(self.names), list(self.columns), self.row_times, self.description).inherit(self)

    def make(self, names: list, columns: list, row_names: Any = None) -> TableObject:
        times = row_names if _is_times(row_names) else self.row_times
        if columns and _height(columns[0]) != times.data.shape[0]:
            return TableObject(names, columns, self.description, None).inherit(self)
        return TimetableObject(names, columns, times, self.description).inherit(self)

    def _sub_row_names(self, rows: np.ndarray) -> Any:
        return _rows(self.row_times, np.asarray(rows, dtype=int).flatten())

    def set_row_names(self, names: Any) -> None:
        if names:
            raise MatlabError("A timetable has row times, not row names.", "MATLAB:timetable:NoRowNames")
        self.row_names = None

    def _pad_row_names(self, n: int) -> None:
        have = self.nrows
        if have < n:
            extra = np.full((n - have, 1), np.nan)
            self.row_times = self.row_times.like(np.vstack([self.row_times.data, extra]))

    # -- selecting rows by time ---------------------------------------------------------------------
    def row_positions(self, idx: Any, for_set: bool = False) -> np.ndarray:
        times = self.row_times.data[:, 0]
        if isinstance(idx, TimeRange):
            return np.nonzero(idx.mask(self.row_times))[0]
        if isinstance(idx, WithTol):
            return np.nonzero(idx.mask(self.row_times))[0]
        if _is_times(idx) or is_text(idx) or isinstance(idx, (Cell, StringArray)):
            wanted = idx
            if not _is_times(idx):
                maker = "datetime" if self.row_times.kind == "datetime" else "duration"
                wanted = BUILTINS[maker](None, [idx], 1)
            if wanted.kind != self.row_times.kind:
                raise MatlabError(f"The row times of this timetable are {self.row_times.kind} values; a "
                                  f"{wanted.kind} cannot select rows.", "MATLAB:timetable:InvalidRowSubscript")
            out: list[int] = []
            for w in wanted.data.flatten(order="F"):
                hits = np.nonzero(times == w)[0]
                if hits.size == 0 and not for_set:
                    raise MatlabError("Unrecognized row time: no row has that time. Use withtol(time, tolerance) "
                                      "or timerange(start, stop) for approximate matches.",
                                      "MATLAB:timetable:UnrecognizedRowTime")
                out += [int(h) for h in hits]
            return np.array(out, dtype=int)
        pos, _ = ix._to_positions(idx, self.nrows, for_set=for_set)
        return pos

    # -- properties -------------------------------------------------------------------------------------
    def time_step(self) -> float:
        d = np.diff(self.row_times.data[:, 0])
        if d.size and np.all(np.abs(d - d[0]) <= 1e-9 * max(1.0, abs(d[0]))):
            return float(d[0])
        return float("nan")

    def properties_struct(self) -> Struct:
        s = super().properties_struct()
        d = s.data.flat[0]
        order = ["Description", "UserData", "DimensionNames", "VariableNames", "VariableDescriptions",
                 "VariableUnits", "VariableContinuity", "RowTimes", "StartTime", "SampleRate", "TimeStep",
                 "CustomProperties"]
        from .datetimes import DurationArray
        step = self.time_step()
        d.pop("RowNames", None)
        d["RowTimes"] = self.row_times
        d["StartTime"] = self.row_times.like(self.row_times.data[:1, :].copy()) if self.nrows else \
            self.row_times.like(np.zeros((0, 0)))
        d["SampleRate"] = np.array([[1.0 / step if step == step and step != 0 else np.nan]])
        d["TimeStep"] = DurationArray(np.array([[step]]), "hh:mm:ss")
        out = Struct()
        for k in order:
            out.set_field(0, k, d[k])
        return out

    def apply_properties(self, value: Struct) -> None:
        fields = value.fields
        old = self.properties_struct().data.flat[0]
        if "RowTimes" in fields:
            new = value.get_field(0, "RowTimes")
            if _is_times(new) and not (_is_times(old["RowTimes"]) and np.array_equal(
                    new.data, old["RowTimes"].data, equal_nan=True) and type(new) is type(old["RowTimes"])):
                col = _time_column(new, "timetable")
                if col.data.shape[0] != self.nrows:
                    raise MatlabError("The number of row times must match the number of rows of the timetable.",
                                      "MATLAB:timetable:RowTimesLengthMismatch")
                self.row_times = col
        if "StartTime" in fields and self.nrows:
            st = value.get_field(0, "StartTime")
            if _is_times(st) and st.data.size == 1 and float(st.data.flat[0]) != float(self.row_times.data[0, 0]):
                shift = float(st.data.flat[0]) - float(self.row_times.data[0, 0])
                base = st if type(st) is type(self.row_times) else self.row_times
                self.row_times = base.like(self.row_times.data + shift)
        for key in ("TimeStep", "SampleRate"):
            if key in fields and self.nrows:
                v = value.get_field(0, key)
                step = float(v.data.flat[0]) if _is_times(v) and v.data.size else (
                    float(to_numeric(v).flat[0]) if isinstance(v, np.ndarray) and v.size else float("nan"))
                if key == "SampleRate":
                    step = 1.0 / step if step == step and step != 0 else float("nan")
                current = self.time_step()
                if step == step and not (current == current and abs(step - current) <= 1e-12 * max(1.0, abs(step))):
                    start = float(self.row_times.data[0, 0])
                    self.row_times = self.row_times.like((start + step * np.arange(self.nrows)).reshape(-1, 1))
        rest = Struct()
        for k in fields:
            if k not in ("RowTimes", "StartTime", "TimeStep", "SampleRate", "RowNames"):
                rest.set_field(0, k, value.get_field(0, k))
        super().apply_properties(rest)

    def get_field(self, interp, name: str) -> Any:
        if name == self.dim_names[0] or (name == "Time" and name not in self.names):
            return self.row_times
        return super().get_field(interp, name)

    def set_field(self, interp, name: str, value: Any) -> TableObject:
        if name == self.dim_names[0] or (name == "Time" and name not in self.names):
            t = self.copy()
            col = _time_column(value, "timetable")
            if col.data.shape[0] != t.nrows:
                raise MatlabError("The number of row times must match the number of rows of the timetable.",
                                  "MATLAB:timetable:RowTimesLengthMismatch")
            t.row_times = col
            return t
        return super().set_field(interp, name, value)

    @staticmethod
    def concat(items: list, axis: int) -> TableObject:
        tts = [t for t in items if not (isinstance(t, np.ndarray) and t.size == 0)]
        if not all(isinstance(t, TimetableObject) for t in tts):
            raise MatlabError("A timetable can only be concatenated with other timetables.",
                              "MATLAB:timetable:InvalidConcatenation")
        first = tts[0]
        if axis == 1:
            names, cols = [], []
            for t in tts:
                if t.nrows != first.nrows or not np.array_equal(t.row_times.data, first.row_times.data,
                                                                equal_nan=True):
                    raise MatlabError("Timetables that are joined side by side must have the same row times; "
                                      "use synchronize to combine timetables with different times.",
                                      "MATLAB:timetable:horzcat:RowTimesMismatch")
                names += t.names
                cols += t.columns
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                raise MatlabError(f"Duplicate table variable name: '{dup[0]}'.", "MATLAB:table:DuplicateVarNames")
            return first.make(names, cols, first.row_times)
        from ..ops import concat as cat
        for t in tts[1:]:
            if set(t.names) != set(first.names):
                raise MatlabError("All tables being vertically concatenated must have the same variable names.",
                                  "MATLAB:table:vertcat:UnequalVarNames")
        cols = [cat([t.columns[t.names.index(n)] for t in tts], 0) for n in first.names]
        times = cat([t.row_times for t in tts], 0)
        return first.make(list(first.names), cols, times)

    # -- display -------------------------------------------------------------------------------------------
    def leading_blocks(self) -> list[tuple[str, list[str], str]]:
        texts = [str(x) for x in np.asarray(self.row_times.element_texts(), dtype=object).flatten(order="F")]
        return [(self.dim_names[0], texts, "center")]

    def display_lines(self) -> list[str]:
        if self.nrows == 0:
            return [self.size_line()]
        return [self.size_line(), ""] + self.body_lines()


def _make(times: Any, names: list[str], cols: list, like: TableObject | None = None) -> TimetableObject:
    tt = TimetableObject(names, cols, times)
    if like is not None:
        tt.inherit(like)
        if not isinstance(like, TimetableObject):
            tt.dim_names = list(TimetableObject.DEFAULT_DIMS)
    return tt


def _need_tt(args: list, name: str, pos: int = 0) -> TimetableObject:
    if len(args) <= pos or not isinstance(args[pos], TimetableObject):
        raise MatlabError(f"{name}: the input must be a timetable.", "MATLAB:timetable:InvalidInput")
    return args[pos]


def _regular_times(opts: dict, n: int, name: str) -> Any:
    """Row times from 'SampleRate' / 'TimeStep' and 'StartTime'."""
    from .datetimes import DurationArray
    if "SampleRate" in opts:
        fs = float(np.real(to_numeric(opts["SampleRate"])).flat[0])
        if not fs > 0:
            raise MatlabError(f"{name}: the sample rate must be positive.", "MATLAB:timetable:InvalidSampleRate")
        step = 1.0 / fs
        step_obj = None
    else:
        ts = opts["TimeStep"]
        if getattr(ts, "kind", "") == "duration":
            step, step_obj = float(ts.data.flat[0]), None
        elif hasattr(ts, "months"):
            step, step_obj = None, ts
        else:
            raise MatlabError(f"{name}: 'TimeStep' must be a duration or calendarDuration.",
                              "MATLAB:timetable:InvalidTimeStep")
    start = opts.get("StartTime")
    if start is not None and not _is_times(start):
        raise MatlabError(f"{name}: 'StartTime' must be a datetime or duration.", "MATLAB:timetable:InvalidStartTime")
    if step_obj is not None:
        if start is None or start.kind != "datetime":
            raise MatlabError(f"{name}: a calendarDuration time step needs a datetime 'StartTime'.",
                              "MATLAB:timetable:InvalidStartTime")
        from .datetimes import add_months
        vals = [float(start.data.flat[0])]
        for k in range(1, n):
            base = float(add_months(np.array(vals[0]), k * float(step_obj.months.flat[0])))
            vals.append(base + k * float(step_obj.days.flat[0]) * DAY + k * float(step_obj.secs.flat[0]))
        return start.like(np.array(vals).reshape(-1, 1)) if n else start.like(np.zeros((0, 1)))
    data = (step * np.arange(n)).reshape(-1, 1)
    if start is None:
        return DurationArray(data, "s" if "SampleRate" in opts else getattr(opts.get("TimeStep"), "fmt", "s"))
    return start.like(float(start.data.flat[0]) + data)


@builtin("timetable")
def _timetable(interp, args, nargout):
    """TT = timetable(rowTimes, var1, ...) or timetable(var1, ..., 'RowTimes' | 'SampleRate' | 'TimeStep', value)."""
    arg_names = list(interp.state.get("call_arg_names") or [])
    rest, opts = _options(list(args), ("VariableNames", "RowTimes", "SampleRate", "TimeStep", "StartTime", "Size",
                                       "VariableTypes", "DimensionNames"), "timetable")
    arg_names = arg_names[:len(rest)]
    time_name = "Time"
    if "Size" in opts:
        h, w = (int(x) for x in np.real(to_numeric(opts["Size"])).flatten()[:2])
        types = _names_from(opts["VariableTypes"]) if "VariableTypes" in opts else ["double"] * w
        base = BUILTINS["table"](interp, [char("Size"), np.array([[float(h), float(w)]]), char("VariableTypes"),
                                         Cell.from_list([char(t) for t in types])], 1)
        cols = list(base.columns)
    else:
        cols = None
    if "RowTimes" in opts:
        times = _time_column(opts["RowTimes"], "timetable")
        data_args, data_names = rest, arg_names
    elif "SampleRate" in opts or "TimeStep" in opts:
        data_args, data_names = rest, arg_names
        n = _height(_as_column(rest[0])) if rest else (_height(cols[0]) if cols else 0)
        times = _regular_times(opts, n, "timetable")
    else:
        if not rest or not _is_times(rest[0]):
            if not rest and cols is None:
                from .datetimes import DatetimeArray
                return TimetableObject([], [], DatetimeArray(np.zeros((0, 1))))
            raise MatlabError("timetable needs row times: give a datetime or duration vector first, or use "
                              "'RowTimes', 'SampleRate' or 'TimeStep'.", "MATLAB:timetable:NoTimeVector")
        times = _time_column(rest[0], "timetable")
        if arg_names and arg_names[0]:
            time_name = arg_names[0]
        data_args, data_names = rest[1:], arg_names[1:]
    if cols is None:
        cols = [_as_column(a) for a in data_args]
    if "VariableNames" in opts:
        names = _names_from(opts["VariableNames"])
    else:
        names = [data_names[k] if k < len(data_names) and data_names[k] else f"Var{k + 1}" for k in range(len(cols))]
    if len(names) != len(cols):
        raise MatlabError("The VariableNames property must contain one name for each variable in the table.",
                          "MATLAB:table:IncorrectNumberOfVarNames")
    for c in cols:
        if _height(c) != times.data.shape[0]:
            raise MatlabError("All the variables must have one row for each row time.",
                              "MATLAB:timetable:RowTimesLengthMismatch")
    tt = TimetableObject(names, cols, times)
    if time_name in names:
        time_name = "Time"
    tt.dim_names = [time_name, "Variables"]
    if "DimensionNames" in opts:
        tt.apply_properties(Struct.scalar({"DimensionNames": opts["DimensionNames"]}))
    return tt


@builtin("istimetable")
def _istimetable(interp, args, nargout):
    nargs(args, 1, 1, "istimetable")
    return bool_scalar(isinstance(args[0], TimetableObject))


@builtin("table2timetable")
def _table2timetable(interp, args, nargout):
    """TT = table2timetable(T[, 'RowTimes', timesOrName | 'SampleRate', fs | 'TimeStep', dt, 'StartTime', t0])."""
    if not args or not isinstance(args[0], TableObject) or isinstance(args[0], TimetableObject):
        raise MatlabError("table2timetable: the input must be a table.", "MATLAB:table2timetable:InvalidInput")
    t = args[0]
    _, opts = _options(list(args[1:]), ("RowTimes", "SampleRate", "TimeStep", "StartTime"), "table2timetable")
    names, cols = list(t.names), list(t.columns)
    time_name = "Time"
    if "RowTimes" in opts:
        rt = opts["RowTimes"]
        if is_text(rt) or (isinstance(rt, np.ndarray) and not is_char(rt)):
            k = t.var_positions(rt)[0]
            time_name = names[k]
            times = _time_column(cols[k], "table2timetable")
            del names[k], cols[k]
        else:
            times = _time_column(rt, "table2timetable")
    elif "SampleRate" in opts or "TimeStep" in opts:
        times = _regular_times(opts, t.nrows, "table2timetable")
    else:
        k = next((j for j, c in enumerate(cols) if _is_times(c)), None)
        if k is None:
            raise MatlabError("table2timetable: the table has no datetime or duration variable to use as row "
                              "times. Give 'RowTimes', 'SampleRate' or 'TimeStep'.",
                              "MATLAB:table2timetable:NoTimeVarFound")
        time_name = names[k]
        times = _time_column(cols[k], "table2timetable")
        del names[k], cols[k]
    if times.data.shape[0] != t.nrows:
        raise MatlabError("The number of row times must match the number of rows of the table.",
                          "MATLAB:timetable:RowTimesLengthMismatch")
    tt = _make(times, names, cols, t)
    tt.dim_names = [time_name, "Variables"]
    return tt


@builtin("timetable2table")
def _timetable2table(interp, args, nargout):
    """T = timetable2table(TT[, 'ConvertRowTimes', tf]): the row times become the first variable."""
    tt = _need_tt(args, "timetable2table")
    _, opts = _options(list(args[1:]), ("ConvertRowTimes",), "timetable2table")
    keep = _flag(opts["ConvertRowTimes"]) if "ConvertRowTimes" in opts else True
    names, cols = list(tt.names), list(tt.columns)
    if keep:
        names, cols = [tt.dim_names[0]] + names, [tt.row_times] + cols
    out = TableObject(names, cols, tt.description)
    out.units, out.var_descriptions = dict(tt.units), dict(tt.var_descriptions)
    out.user_data = tt.user_data
    return out


@builtin("array2timetable")
def _array2timetable(interp, args, nargout):
    """TT = array2timetable(A, 'RowTimes', t | 'SampleRate', fs | 'TimeStep', dt, 'VariableNames', names)."""
    rest, opts = _options(list(args), ("RowTimes", "SampleRate", "TimeStep", "StartTime", "VariableNames",
                                       "DimensionNames"), "array2timetable", 1)
    nargs(rest, 1, 1, "array2timetable")
    extra = [char("VariableNames"), opts["VariableNames"]] if "VariableNames" in opts else []
    t = BUILTINS["array2table"](interp, [rest[0], *extra], 1)
    pass_on = []
    for k in ("RowTimes", "SampleRate", "TimeStep", "StartTime"):
        if k in opts:
            pass_on += [char(k), opts[k]]
    if not pass_on:
        raise MatlabError("array2timetable needs 'RowTimes', 'SampleRate' or 'TimeStep'.",
                          "MATLAB:array2timetable:NoTimeVector")
    tt = _table2timetable(interp, [t, *pass_on], 1)
    if "DimensionNames" in opts:
        tt.apply_properties(Struct.scalar({"DimensionNames": opts["DimensionNames"]}))
    return tt


# -- time subscripts ------------------------------------------------------------------------------------------

class TimeRange(MObject):
    """timerange(start, stop, type): selects the rows whose times fall in an interval."""

    def __init__(self, start: Any, stop: Any, kind: str):
        self.start, self.stop, self.kind = start, stop, kind

    def class_name(self) -> str:
        return "timerange"

    def isa_class(self, name: str) -> bool:
        return name == "timerange"

    def mask(self, times: Any) -> np.ndarray:
        if times.kind != self.start.kind:
            raise MatlabError(f"The row times are {times.kind} values but the time range was made from "
                              f"{self.start.kind} values.", "MATLAB:timerange:MismatchRowTimesType")
        t = times.data[:, 0]
        lo, hi = float(self.start.data.flat[0]), float(self.stop.data.flat[0])
        left = t >= lo if self.kind in ("openright", "closedleft", "closed") else t > lo
        right = t <= hi if self.kind in ("openleft", "closedright", "closed") else t < hi
        return left & right

    def display_lines(self) -> list[str]:
        a = str(self.start.element_texts().flat[0])
        b = str(self.stop.element_texts().flat[0])
        lb = "[" if self.kind in ("openright", "closedleft", "closed") else "("
        rb = "]" if self.kind in ("openleft", "closedright", "closed") else ")"
        return ["  timetable timerange subscript:", "", "    Select timetable rows with times in the half-open "
                "interval:" if self.kind in ("openright", "closedleft") else "    Select timetable rows with "
                "times in the interval:", f"      {lb}{a}, {b}{rb}"]

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


@builtin("timerange")
def _timerange(interp, args, nargout):
    """S = timerange(start, stop[, intervalType | unit])."""
    nargs(args, 2, 3, "timerange")

    def as_time(v: Any) -> Any:
        if _is_times(v):
            return v
        if is_text(v):
            word = text(v).strip().lower()
            if word in ("-inf", "inf"):
                from .datetimes import DatetimeArray
                return DatetimeArray(np.array([[np.inf if word == "inf" else -np.inf]]))
            return BUILTINS["datetime"](interp, [v], 1)
        if isinstance(v, np.ndarray) and v.size == 1 and not np.isfinite(float(np.real(v).flat[0])):
            from .datetimes import DatetimeArray
            return DatetimeArray(np.array([[float(np.real(v).flat[0])]]))
        raise MatlabError("timerange: the limits must be datetime or duration values, or date text.",
                          "MATLAB:timerange:InvalidLimits")
    start, stop = as_time(args[0]), as_time(args[1])
    if np.isinf(start.data.flat[0]) and not np.isinf(stop.data.flat[0]):
        start = stop.like(start.data)
    if np.isinf(stop.data.flat[0]) and not np.isinf(start.data.flat[0]):
        stop = start.like(stop.data)
    kind = "openright"
    if len(args) > 2:
        word = text(args[2]).lower()
        if word in ("openright", "closedleft", "openleft", "closedright", "open", "closed"):
            kind = word
        else:
            unit = word.rstrip("s")
            if unit not in ("year", "quarter", "month", "week", "day", "hour", "minute", "second") \
                    or start.kind != "datetime":
                raise MatlabError(f"timerange: unknown interval type or unit '{text(args[2])}'.",
                                  "MATLAB:timerange:InvalidIntervalType")
            shift = BUILTINS["dateshift"]
            start = shift(interp, [start, char("start"), char(unit)], 1)
            stop = shift(interp, [stop, char("start"), char(unit), char("next")], 1)
    if start.kind != stop.kind:
        raise MatlabError("timerange: the start and the end must both be datetimes or both be durations.",
                          "MATLAB:timerange:MismatchLimitTypes")
    return TimeRange(start, stop, kind)


class WithTol(MObject):
    """withtol(times, tol): selects the rows whose times are within a tolerance of the given times."""

    def __init__(self, times: Any, tol: float):
        self.times, self.tol = times, tol

    def class_name(self) -> str:
        return "withtol"

    def isa_class(self, name: str) -> bool:
        return name == "withtol"

    def mask(self, times: Any) -> np.ndarray:
        if times.kind != self.times.kind:
            raise MatlabError(f"The row times are {times.kind} values but the tolerance subscript was made "
                              f"from {self.times.kind} values.", "MATLAB:withtol:MismatchRowTimesType")
        t = times.data[:, 0]
        out = np.zeros(t.shape, dtype=bool)
        for w in self.times.data.flatten(order="F"):
            out |= np.abs(t - w) <= self.tol
        return out

    def display_lines(self) -> list[str]:
        return ["  timetable withtol subscript:", "",
                f"    Select timetable rows matching the specified times within {self.tol:g} sec"]

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


@builtin("withtol")
def _withtol(interp, args, nargout):
    """S = withtol(rowTimes, tol)."""
    nargs(args, 2, 2, "withtol")
    times = args[0]
    if not _is_times(times):
        times = BUILTINS["datetime"](interp, [times], 1)
    tol = args[1]
    if getattr(tol, "kind", "") != "duration":
        raise MatlabError("withtol: the tolerance must be a duration.", "MATLAB:withtol:InvalidTolerance")
    return WithTol(times, abs(float(tol.data.flat[0])))


# -- regularity, lag ----------------------------------------------------------------------------------------------

@builtin("isregular")
def _isregular(interp, args, nargout):
    """[tf, dt] = isregular(TT[, timeComponent])."""
    tt = args[0] if args else None
    if isinstance(tt, TimetableObject):
        times = tt.row_times
    elif _is_times(tt):
        times = tt
    else:
        raise MatlabError("isregular: the input must be a timetable, or a datetime or duration vector.",
                          "MATLAB:isregular:InvalidInput")
    from .datetimes import CalendarDuration, DurationArray, fields_from_seconds
    unit = text(args[1]).lower().rstrip("s") if len(args) > 1 else "time"
    data = times.data.flatten(order="F")
    d = np.diff(data)
    if unit == "time":
        ok = d.size > 0 and bool(np.all(np.abs(d - d[0]) <= 1e-9 * max(1.0, abs(d[0])))) and not np.any(np.isnan(d))
        step = DurationArray(np.array([[d[0] if ok else np.nan]]), "hh:mm:ss")
        return (bool_scalar(ok), step) if nargout > 1 else bool_scalar(ok)
    if times.kind != "datetime" or unit not in ("year", "quarter", "month", "week", "day"):
        raise MatlabError("isregular: the time component must be 'years', 'quarters', 'months', 'weeks', 'days' "
                          "or 'time' (calendar units need datetime row times).",
                          "MATLAB:isregular:InvalidTimeComponent")
    y, mo, dd, hh, mi, ss = (np.asarray(x, dtype=float) for x in fields_from_seconds(data))
    months = y * 12 + mo
    if unit in ("year", "quarter", "month"):
        per = {"year": 12, "quarter": 3, "month": 1}[unit]
        dm = np.diff(months)
        rest_same = bool(np.all(np.diff(dd) == 0) and np.all(np.diff(hh * 3600 + mi * 60 + ss) == 0))
        ok = dm.size > 0 and bool(np.all(dm == dm[0])) and dm[0] % per == 0 and rest_same
        step = CalendarDuration(float(dm[0]) if ok else np.nan, 0.0, 0.0)
    else:
        per = 7.0 if unit == "week" else 1.0
        days = d / DAY
        ok = d.size > 0 and bool(np.all(days == days[0])) and days[0] % per == 0
        step = CalendarDuration(0.0, float(days[0]) if ok else np.nan, 0.0)
    return (bool_scalar(ok), step) if nargout > 1 else bool_scalar(ok)


def _shift_column(col: Any, n: int) -> Any:
    """Move the data of a variable down by n rows (up for negative n), filling with missing values."""
    h = _height(col)
    if n == 0:
        return col
    src = np.arange(h) - n
    valid = (src >= 0) & (src < h)
    if isinstance(col, (Cell, StringArray)):
        data = np.empty(col.data.shape, dtype=object)
        for i in range(h):
            if valid[i]:
                data[i] = col.data[src[i]]
            elif isinstance(col, Cell):
                for idx in np.ndindex(data[i].shape):
                    data[i][idx] = char("") if all(is_char(x) for x in col.data.flat) else np.zeros((0, 0))
            else:
                data[i] = None
        return type(col)(data)
    if getattr(col, "kind", None) in ("datetime", "duration", "categorical"):
        out = np.full(col.data.shape, np.nan)
        out[valid] = col.data[src[valid]]
        return col.like(out)
    if isinstance(col, np.ndarray) and col.dtype.kind in "fc":
        out = np.full(col.shape, np.nan, dtype=col.dtype)
        out[valid] = col[src[valid]]
        return out
    if isinstance(col, np.ndarray) and col.dtype.kind in "biu":
        out = np.zeros(col.shape, dtype=col.dtype)
        out[valid] = col[src[valid]]
        return out
    raise MatlabError("lag: a variable of this type cannot be shifted.", "MATLAB:timetable:lag:UnsupportedType")


@builtin("lag")
def _lag(interp, args, nargout):
    """TT2 = lag(TT[, n | dt]): shift the data forward in time; the row times stay."""
    tt = _need_tt(args, "lag")
    step = tt.time_step()
    if step != step:
        raise MatlabError("lag: the timetable must be regular (the same time step between all rows). Use "
                          "retime first.", "MATLAB:timetable:lag:IrregularTimetable")
    n = 1
    if len(args) > 1:
        v = args[1]
        if getattr(v, "kind", "") == "duration":
            ratio = float(v.data.flat[0]) / step
            if abs(ratio - round(ratio)) > 1e-9:
                raise MatlabError("lag: the shift must be a whole multiple of the time step.",
                                  "MATLAB:timetable:lag:NonMultipleShift")
            n = int(round(ratio))
        else:
            x = float(np.real(to_numeric(v)).flat[0])
            if x != int(x):
                raise MatlabError("lag: the number of time steps must be a whole number.",
                                  "MATLAB:timetable:lag:InvalidShift")
            n = int(x)
    return tt.make(list(tt.names), [_shift_column(c, n) for c in tt.columns], tt.row_times)


# -- retime / synchronize ---------------------------------------------------------------------------------------------

def _unit_starts(interp, times: Any, unit: str) -> Any:
    """Starts of the units (days, months ...) that cover the row times."""
    if times.kind != "datetime":
        raise MatlabError("Calendar time steps such as 'daily' need datetime row times.",
                          "MATLAB:timetable:retime:InvalidTimeStep")
    data = times.data[~np.isnan(times.data)]
    if data.size == 0:
        return times.like(np.zeros((0, 1)))
    shift = BUILTINS["dateshift"]
    first = shift(interp, [times.like(np.array([[float(data.min())]])), char("start"), char(unit)], 1)
    start, last = float(first.data.flat[0]), float(data.max())
    fixed = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": DAY, "week": 7 * DAY}.get(unit)
    if fixed is not None and getattr(times, "zone", "") in ("", "UTC"):
        count = int(np.floor((last - start) / fixed + 1e-9)) + 1
        return times.like((start + fixed * np.arange(count)).reshape(-1, 1))
    out = [start]
    k = 1
    while len(out) < 200000:
        nxt = shift(interp, [first, char("start"), char(unit), np.array([[float(k)]])], 1)
        v = float(nxt.data.flat[0])
        if v > last:
            break
        out.append(v)
        k += 1
    return times.like(np.array(out).reshape(-1, 1))


def _bin_edges(interp, new_times: Any, unit: Any) -> np.ndarray:
    """Right edges of the aggregation bins that start at the new row times."""
    t = new_times.data[:, 0]
    if t.size == 0:
        return t
    if isinstance(unit, float):  # a regular time step: every bin is one step long
        return np.append(t[1:], t[-1] + unit)
    if unit is not None:
        last = BUILTINS["dateshift"](interp, [new_times.like(t[-1:].reshape(1, 1)), char("start"), char(unit),
                                             char("next")], 1)
        return np.append(t[1:], float(last.data.flat[0]))
    # explicit new times: the last bin holds only the data at exactly the last time
    return np.append(t[1:], np.nextafter(t[-1], np.inf))


def _numeric(col: Any, method: str) -> np.ndarray:
    if getattr(col, "kind", None) in ("duration", "datetime"):
        return col.data.astype(float)
    if isinstance(col, np.ndarray) and col.dtype.kind in "biuf":
        return col.astype(float)
    raise MatlabError(f"The method '{method}' needs numeric data; a variable of this type cannot be "
                      "interpolated or averaged. Use 'previous', 'next', 'nearest' or 'fillwithmissing', or "
                      "select the numeric variables first.", "MATLAB:timetable:retime:NonNumericData")


def _wrap_like(col: Any, data: np.ndarray) -> Any:
    if getattr(col, "kind", None) in ("duration", "datetime"):
        return col.like(data)
    return data


def _pick_rows(col: Any, src: np.ndarray) -> Any:
    """Rows of a variable by source row number, where -1 gives a missing value."""
    ok = src >= 0
    if isinstance(col, (Cell, StringArray)):
        shape = (src.size,) + col.data.shape[1:]
        data = np.empty(shape, dtype=object)
        cellstr = isinstance(col, Cell) and all(is_char(x) for x in col.data.flat)
        for i, s in enumerate(src):
            if s >= 0:
                data[i] = col.data[s]
            elif isinstance(col, Cell):
                for idx in np.ndindex(shape[1:]):
                    data[(i,) + idx] = char("") if cellstr else np.zeros((0, 0))
            else:
                data[i] = None
        return type(col)(data)
    if getattr(col, "kind", None) in ("datetime", "duration", "categorical"):
        out = np.full((src.size,) + col.data.shape[1:], np.nan)
        out[ok] = col.data[src[ok]]
        return col.like(out)
    if isinstance(col, np.ndarray):
        if col.dtype.kind in "biu" and not ok.all():
            col = col.astype(float)
        fill = np.nan if col.dtype.kind in "fc" else (" " if col.dtype.kind == "U" else 0)
        out = np.full((src.size,) + col.shape[1:], fill, dtype=col.dtype)
        out[ok] = col[src[ok]]
        return out
    raise MatlabError("A variable of this type cannot be moved to new row times.",
                      "MATLAB:timetable:retime:UnsupportedType")


def _retime_column(interp, col: Any, old: np.ndarray, new: np.ndarray, method: Any, edges: np.ndarray | None,
                   opts: dict) -> Any:
    n = new.size
    order = np.argsort(old, kind="stable")
    ts = old[order]
    if isinstance(method, FunctionHandle) or method in _AGG:
        values = []
        for k in range(n):
            lo, hi = new[k], edges[k]
            rows = order[(ts >= lo) & (ts < hi)]
            piece = _rows(col, rows)
            if isinstance(method, FunctionHandle):
                values.append(interp.call_function_value(method, [piece], 1)[0])
                continue
            if method == "count":
                miss = _missing_mask(piece)
                values.append(np.array([[float(np.sum(~miss.reshape(rows.size, -1).any(axis=1)))]]) if rows.size
                              else np.array([[0.0]]))
                continue
            if method in ("firstvalue", "lastvalue"):
                if rows.size == 0:
                    values.append(None)
                else:
                    values.append(_rows(col, rows[:1] if method == "firstvalue" else rows[-1:]))
                continue
            a = _numeric(piece, method)
            width = a.shape[1] if a.ndim > 1 else 1
            if rows.size == 0:
                empty = 0.0 if method == "sum" else (1.0 if method == "prod" else np.nan)
                values.append(np.full((1, width), empty))
                continue
            import warnings
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore")
                fn = {"sum": np.nansum, "mean": np.nanmean, "prod": np.nanprod, "min": np.nanmin,
                      "max": np.nanmax, "median": np.nanmedian, "std": lambda x, axis: np.nanstd(x, axis=axis, ddof=1)
                      if x.shape[0] > 1 else np.zeros(x.shape[1]), "var": lambda x, axis: np.nanvar(
                          x, axis=axis, ddof=1) if x.shape[0] > 1 else np.zeros(x.shape[1]),
                      "mode": lambda x, axis: np.array([_mode(x[:, j]) for j in range(x.shape[1])])}[method]
                values.append(np.asarray(fn(a, axis=0), dtype=float).reshape(1, -1))
        if not isinstance(method, FunctionHandle) and method in ("firstvalue", "lastvalue"):
            src = np.array([-1 if v is None else 0 for v in values])
            if (src >= 0).all() and values:
                from ..ops import concat
                return concat(values, 0)
            picked = []
            for k in range(n):
                rows = order[(ts >= new[k]) & (ts < edges[k])]
                picked.append(-1 if rows.size == 0 else int(rows[0] if method == "firstvalue" else rows[-1]))
            return _pick_rows(col, np.array(picked, dtype=int))
        if isinstance(method, FunctionHandle):
            return _column_from_values(values)
        stacked = np.vstack(values) if values else np.zeros((0, 1))
        return _wrap_like(col, stacked) if method not in ("count",) else stacked
    if method in _INTERP:
        a = _numeric(col, method)[order]
        a = a.reshape(a.shape[0], -1)
        out = np.full((n, a.shape[1]), np.nan)
        extrap = opts.get("EndValues")
        const_end = None
        if extrap is not None and not (is_text(extrap) and text(extrap).lower() == "extrap"):
            const_end = float(np.real(to_numeric(extrap)).flat[0])
        for j in range(a.shape[1]):
            ok = ~np.isnan(a[:, j]) & ~np.isnan(ts)
            x, y = ts[ok], a[ok, j]
            if x.size == 0:
                continue
            ux, first = np.unique(x, return_index=True)
            uy = y[first]
            if ux.size == 1:
                out[:, j] = np.where(new == ux[0], uy[0], np.nan)
                continue
            if method == "linear":
                v = np.interp(new, ux, uy)
                left, right = new < ux[0], new > ux[-1]
                v[left] = uy[0] + (uy[1] - uy[0]) / (ux[1] - ux[0]) * (new[left] - ux[0])
                v[right] = uy[-1] + (uy[-1] - uy[-2]) / (ux[-1] - ux[-2]) * (new[right] - ux[-1])
            else:
                from scipy import interpolate
                base = float(ux[0])
                xs, xn = ux - base, new - base
                if method == "spline":
                    v = interpolate.CubicSpline(xs, uy, bc_type="not-a-knot")(xn) if ux.size > 2 else \
                        np.interp(xn, xs, uy)
                elif method == "pchip":
                    v = interpolate.PchipInterpolator(xs, uy, extrapolate=True)(xn)
                else:
                    v = interpolate.Akima1DInterpolator(xs, uy, method="makima")(xn, extrapolate=True) \
                        if ux.size > 2 else np.interp(xn, xs, uy)
            if const_end is not None:
                v = np.where((new < ux[0]) | (new > ux[-1]), const_end, v)
            out[:, j] = v
        shaped = out.reshape((n,) + (col.data.shape[1:] if hasattr(col, "data") and not isinstance(
            col, np.ndarray) else col.shape[1:]))
        return _wrap_like(col, shaped)
    # fill methods: every new time takes one source row (or none)
    src = np.full(n, -1, dtype=int)
    if ts.size:
        if method in ("fillwithmissing", "fillwithconstant", "default"):
            pos = np.searchsorted(ts, new, side="left")
            hit = (pos < ts.size) & (ts[np.minimum(pos, ts.size - 1)] == new)
            src[hit] = order[pos[hit]]
        elif method == "previous":
            pos = np.searchsorted(ts, new, side="right") - 1
            src[pos >= 0] = order[pos[pos >= 0]]
        elif method == "next":
            pos = np.searchsorted(ts, new, side="left")
            src[pos < ts.size] = order[pos[pos < ts.size]]
        elif method == "nearest":
            pos = np.searchsorted(ts, new, side="left")
            lo = np.clip(pos - 1, 0, ts.size - 1)
            hi = np.clip(pos, 0, ts.size - 1)
            pick = np.where(np.abs(new - ts[lo]) < np.abs(ts[hi] - new), lo, hi)
            src = order[pick]
    out = _pick_rows(col, src)
    if method == "fillwithconstant":
        if "Constant" not in opts:
            raise MatlabError("The method 'fillwithconstant' needs the 'Constant' option.",
                              "MATLAB:timetable:retime:MissingConstant")
        const = opts["Constant"]
        miss = src < 0
        if miss.any():
            if isinstance(out, np.ndarray) and not is_char(out):
                out = out.copy()
                out[miss] = float(np.real(to_numeric(const)).flat[0])
            elif isinstance(out, StringArray):
                out.data[miss] = str(const) if isinstance(const, MString) else text(const)
            elif isinstance(out, Cell):
                for i in np.nonzero(miss)[0]:
                    for idx in np.ndindex(out.data[i].shape):
                        out.data[(i,) + idx] = const
    return out


def _mode(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    vals, counts = np.unique(x, return_counts=True)
    return float(vals[np.argmax(counts)])


def _method_of(v: Any, name: str) -> Any:
    if isinstance(v, FunctionHandle):
        return v
    word = text(v).lower()
    if word not in _FILL + _INTERP + _AGG:
        raise MatlabError(f"{name}: unknown method '{text(v)}'. Use 'fillwithmissing', 'fillwithconstant', "
                          "'previous', 'next', 'nearest', 'linear', 'spline', 'pchip', 'makima', 'sum', 'mean', "
                          "'prod', 'min', 'max', 'count', 'firstvalue', 'lastvalue' or a function handle.",
                          "MATLAB:timetable:retime:InvalidMethod")
    return word


def _new_times(interp, sources: list[TimetableObject], spec: Any, opts: dict, name: str
               ) -> tuple[Any, Any]:
    """(new row times, calendar unit of the step if there is one) for retime / synchronize."""
    ref = sources[0].row_times
    all_data = np.concatenate([t.row_times.data[:, 0] for t in sources]) if sources else np.zeros(0)
    all_times = ref.like(all_data.reshape(-1, 1))
    if _is_times(spec):
        if spec.kind != ref.kind:
            raise MatlabError(f"{name}: the new row times must be {ref.kind} values like the row times of the "
                              "timetable.", "MATLAB:timetable:retime:MismatchRowTimesType")
        return ref.like(np.sort(spec.data.flatten(order="F")).reshape(-1, 1)), None
    word = text(spec).lower()
    if word in _STEPS:
        return _unit_starts(interp, all_times, _STEPS[word]), _STEPS[word]
    if word == "regular":
        if "TimeStep" not in opts and "SampleRate" not in opts:
            raise MatlabError(f"{name}: 'regular' needs the 'TimeStep' or 'SampleRate' option.",
                              "MATLAB:timetable:retime:MissingTimeStep")
        lo, hi = float(np.nanmin(all_data)), float(np.nanmax(all_data))
        if "SampleRate" in opts:
            step = 1.0 / float(np.real(to_numeric(opts["SampleRate"])).flat[0])
        else:
            ts = opts["TimeStep"]
            if getattr(ts, "kind", "") != "duration":
                if hasattr(ts, "months") and ref.kind == "datetime":
                    count = 0
                    out = []
                    from .datetimes import add_months
                    while len(out) < 200000:
                        v = float(add_months(np.array(lo), count * float(ts.months.flat[0]))) \
                            + count * float(ts.days.flat[0]) * DAY + count * float(ts.secs.flat[0])
                        if v > hi:
                            break
                        out.append(v)
                        count += 1
                    return ref.like(np.array(out).reshape(-1, 1)), None
                raise MatlabError(f"{name}: 'TimeStep' must be a duration or calendarDuration.",
                                  "MATLAB:timetable:InvalidTimeStep")
            step = float(ts.data.flat[0])
        if not step > 0:
            raise MatlabError(f"{name}: the time step must be positive.", "MATLAB:timetable:InvalidTimeStep")
        count = int(np.floor((hi - lo) / step + 1e-9)) + 1
        return ref.like((lo + step * np.arange(count)).reshape(-1, 1)), float(step)
    if word in ("union", "intersection", "commonrange", "first", "last"):
        sets = [np.unique(t.row_times.data[:, 0][~np.isnan(t.row_times.data[:, 0])]) for t in sources]
        if word == "union":
            vals = np.unique(np.concatenate(sets)) if sets else np.zeros(0)
        elif word == "intersection":
            vals = sets[0]
            for s in sets[1:]:
                vals = np.intersect1d(vals, s)
        elif word == "commonrange":
            lo = max(s.min() for s in sets if s.size)
            hi = min(s.max() for s in sets if s.size)
            vals = np.unique(np.concatenate(sets))
            vals = vals[(vals >= lo) & (vals <= hi)]
        elif word == "first":
            vals = sets[0]
        else:
            vals = sets[-1]
        return ref.like(vals.reshape(-1, 1)), None
    raise MatlabError(f"{name}: unknown time basis '{text(spec)}'.", "MATLAB:timetable:retime:InvalidTimeBasis")


def _is_time_spec(v: Any) -> bool:
    return _is_times(v) or (is_text(v) and text(v).lower() in tuple(_STEPS) + (
        "regular", "union", "intersection", "commonrange", "first", "last"))


def _retime_table(interp, tt: TimetableObject, new: Any, unit: Any, method: Any, opts: dict) -> list:
    old = tt.row_times.data[:, 0]
    target = new.data[:, 0]
    agg = isinstance(method, FunctionHandle) or method in _AGG
    edges = _bin_edges(interp, new, unit) if agg else None
    return [_retime_column(interp, c, old, target, method, edges, opts) for c in tt.columns]


@builtin("retime")
def _retime(interp, args, nargout):
    """TT2 = retime(TT, newTimeStep | newTimes | 'regular', method, ...)."""
    tt = _need_tt(args, "retime")
    rest, opts = _options(list(args[1:]), ("TimeStep", "SampleRate", "Constant", "EndValues",
                                           "IncludedEdge"), "retime", 1)
    if not rest:
        raise MatlabError("retime needs the new row times or a time step.", "MATLAB:minrhs")
    method = _method_of(rest[1], "retime") if len(rest) > 1 else "fillwithmissing"
    new, unit = _new_times(interp, [tt], rest[0], opts, "retime")
    cols = _retime_table(interp, tt, new, unit, method, opts)
    return tt.make(list(tt.names), cols, new)


@builtin("synchronize")
def _synchronize(interp, args, nargout):
    """TT = synchronize(TT1, TT2, ...[, newTimeBasis | newTimeStep | newTimes][, method], ...)."""
    arg_names = list(interp.state.get("call_arg_names") or [])
    rest, opts = _options(list(args), ("TimeStep", "SampleRate", "Constant", "EndValues", "IncludedEdge"),
                          "synchronize", 1)
    tables = []
    while rest and isinstance(rest[0], TableObject):
        t = rest.pop(0)
        if not isinstance(t, TimetableObject):
            raise MatlabError("synchronize: every table must be a timetable; convert tables with "
                              "table2timetable.", "MATLAB:timetable:synchronize:NotTimetable")
        tables.append(t)
    if not tables:
        raise MatlabError("synchronize needs at least one timetable.", "MATLAB:minrhs")
    spec: Any = char("union")
    method: Any = "fillwithmissing"
    if rest and _is_time_spec(rest[0]):
        spec = rest.pop(0)
    if rest:
        method = _method_of(rest.pop(0), "synchronize")
    if rest:
        raise MatlabError("synchronize: unexpected extra arguments.", "MATLAB:timetable:synchronize:TooManyInputs")
    new, unit = _new_times(interp, tables, spec, opts, "synchronize")
    counts: dict[str, int] = {}
    for t in tables:
        for n in t.names:
            counts[n] = counts.get(n, 0) + 1
    names, cols = [], []
    for k, t in enumerate(tables):
        label = arg_names[k] if k < len(arg_names) and arg_names[k] else str(k + 1)
        for n, c in zip(t.names, _retime_table(interp, t, new, unit, method, opts)):
            names.append(f"{n}_{label}" if counts[n] > 1 else n)
            cols.append(c)
    dup = [x for x in names if names.count(x) > 1]
    if dup:
        raise MatlabError(f"synchronize: the result would have two variables called '{dup[0]}'.",
                          "MATLAB:table:DuplicateVarNames")
    return tables[0].make(names, cols, new)


# -- table functions that keep the row times ---------------------------------------------------------------------------

def _sortrows_tt(interp, args, nargout):
    tt = args[0]
    rest = list(args[1:])
    by_time = not rest or (is_text(rest[0]) and text(rest[0]) in (tt.dim_names[0], "Time") and
                           text(rest[0]) not in tt.names) or (is_text(rest[0]) and text(rest[0]).lower() in
                                                              ("ascend", "descend"))
    if not by_time:
        return _TABLE_SORT(interp, args, nargout)
    words = [text(a).lower() for a in rest if is_text(a)]
    descend = "descend" in words
    t = tt.row_times.data[:, 0]
    key = np.where(np.isnan(t), np.inf, t)
    order = np.argsort(-key if descend else key, kind="stable")
    res = tt.index_get(interp, [order.reshape(-1, 1) + 1.0, ix.COLON])
    return (res, (order + 1.0).reshape(-1, 1)) if nargout > 1 else res


_TABLE_SORT = _TABLE_FUNCS["sortrows"]


def _dispatch_sortrows(interp, args, nargout):
    if args and isinstance(args[0], TimetableObject):
        return _sortrows_tt(interp, args, nargout)
    return _TABLE_SORT(interp, args, nargout)


_TABLE_FUNCS["sortrows"] = _dispatch_sortrows


def _issorted_tt(interp, args, nargout):
    tt = args[0]
    if not isinstance(tt, TimetableObject):
        return BUILTINS["issorted"](interp, [tt.to_array(), *args[1:]], nargout)
    t = tt.row_times.data[:, 0]
    return bool_scalar(bool(np.all(np.diff(t) >= 0)) if t.size > 1 else True)


TableObject._METHODS = TableObject._METHODS | {"issorted"}
_TABLE_FUNCS["issorted"] = _issorted_tt
