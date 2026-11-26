"""timeseries: samples of data tied to a vector of times, and the methods that work on them.

How the data is laid out (the same way MATLAB code sees ``ts.Data``):

* data with one or two dimensions whose first dimension has one row per time runs along its first
  dimension (``IsTimeFirst`` is true);
* data with three or more dimensions runs along its last dimension, and so does a matrix whose
  columns (not rows) match the times: it is stored as rows-by-1-by-times (``IsTimeFirst`` false).

Internally the data always keeps its time dimension (a time-last series with one sample is
rows-by-columns-by-1); ``ts.Data`` drops the trailing singleton the way MATLAB arrays do.

``ts.Time`` holds times relative to ``TimeInfo.StartDate`` (or plain numbers when there is no start
date) in ``TimeInfo.Units``. ``TimeInfo``, ``DataInfo``, ``QualityInfo``, the interpolation method
and events are small value objects of their own (``tsdata.timemetadata`` and friends).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import ops
from ..display import size_str, summary
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, as_2d, char, class_name, is_char, to_str,
)
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, fnum, is_text, match_option, text

UNITS = {"weeks": 604800.0, "days": 86400.0, "hours": 3600.0, "minutes": 60.0, "seconds": 1.0,
         "milliseconds": 1e-3, "microseconds": 1e-6, "nanoseconds": 1e-9}
PROPS = ["Events", "Name", "UserData", "Data", "DataInfo", "Time", "TimeInfo", "Quality", "QualityInfo",
         "IsTimeFirst", "TreatNaNasMissing", "Length"]
READ_ONLY = {"IsTimeFirst", "Length"}
DATE_FORMAT = "dd-mmm-yyyy HH:MM:SS"
TOL = 1e-10
EPOCH_DATENUM = 719529.0  # datenum of 1970-01-01


def _err(msg: str, ident: str = "MATLAB:timeseries:InvalidInput") -> MatlabError:
    return MatlabError(msg, ident)


def _resolve(name: str, names, what: str = "property", cls: str = "timeseries") -> str:
    """The entry of ``names`` that ``name`` refers to (ignoring case), or an error."""
    if name in names:
        return name
    low = name.lower()
    for n in names:
        if n.lower() == low:
            return n
    raise MatlabError(f"Unrecognized {what} '{name}' for class '{cls}'.", "MATLAB:noSuchMethodOrField")


def _units(v: Any) -> str:
    u = text(v, "time units")
    key = match_option(u, UNITS, partial=False)
    if key is None:
        raise _err(f"'{u}' is not a time unit. Use one of: " + ", ".join(UNITS) + ".",
                   "MATLAB:timeseries:InvalidUnits")
    return key


def _empty() -> np.ndarray:
    return np.zeros((0, 0))


def _is_empty_value(v: Any) -> bool:
    return (isinstance(v, np.ndarray) and v.size == 0) or (isinstance(v, Cell) and v.data.size == 0)


def _scalar(v: Any, what: str) -> float:
    a = fnum(v)
    if a.size != 1 or np.iscomplexobj(a):
        raise _err(f"The {what} must be a real scalar.")
    return float(a.reshape(-1)[0])


def _flag(v: Any, what: str) -> bool:
    if is_text(v):
        s = text(v).lower()
        if s in ("on", "true"):
            return True
        if s in ("off", "false"):
            return False
    a = np.asarray(v) if isinstance(v, np.ndarray) else None
    if a is None or a.size != 1 or a.dtype.kind not in "biuf":
        raise _err(f"The {what} must be true or false.")
    return bool(a.reshape(-1)[0])


def _name_value(args: list, known, fname: str) -> dict:
    """Name-value pairs (names matched ignoring case, a unique prefix is enough)."""
    if len(args) % 2:
        raise _err(f"{fname}: options must come in name-value pairs.", f"MATLAB:{fname}:InvalidOption")
    out = {}
    for k in range(0, len(args), 2):
        if not is_text(args[k]):
            raise _err(f"{fname}: expected an option name, found a value of class '{class_name(args[k])}'.",
                       f"MATLAB:{fname}:InvalidOption")
        key = match_option(text(args[k]), known)
        if key is None:
            raise _err(f"{fname}: unknown option '{text(args[k])}'. The options are: " + ", ".join(known) + ".",
                       f"MATLAB:{fname}:InvalidOption")
        out[key] = args[k + 1]
    return out


# -- dates ----------------------------------------------------------------------------------------------

def _texts(v: Any) -> list[str] | None:
    if isinstance(v, MString):
        return [str(v)]
    if isinstance(v, StringArray):
        return ["" if x is None else x for x in v.values_f()]
    if isinstance(v, Cell) and all(is_text(x) for x in v.data.flat):
        return [to_str(x) for x in v.values_f()]
    if is_char(v):
        a = v if v.ndim == 2 else v.reshape(1, -1)
        return ["".join(r).strip() for r in a] if a.shape[0] > 1 else [to_str(v)]
    return None


def _is_datetime(v: Any) -> bool:
    return getattr(v, "kind", None) == "datetime"


def _datenums(interp, v: Any) -> np.ndarray:
    """Days since 1970-01-01 of date texts or datetime values (a 1-D array).

    Counting from 1970 rather than from year 0 keeps the times precise to well under a microsecond."""
    if _is_datetime(v):
        return np.asarray(v.data, dtype=float).reshape(-1, order="F") / 86400.0
    items = _texts(v)
    if items is None:
        raise _err("Dates must be given as text (a character vector, a cell array of them or a string array) "
                   "or as datetime values.")
    from .datetimes import parse_texts
    out = np.empty(len(items))
    for k, s in enumerate(items):
        try:
            secs, _ = parse_texts([s], None)
            val = float(secs.reshape(-1)[0]) / 86400.0
        except MatlabError:
            val = float("nan")
        if not np.isfinite(val):
            try:
                val = float(np.asarray(BUILTINS["datenum"](interp, [char(s)], 1)).reshape(-1)[0]) - EPOCH_DATENUM
            except MatlabError:
                pass
        if not np.isfinite(val):
            raise _err(f"Unable to read '{s}' as a date.", "MATLAB:timeseries:InvalidDate")
        out[k] = val
    return out


def _datestr(interp, days: float, fmt: str = "") -> str:
    """Date text for a number of days since 1970-01-01."""
    return to_str(BUILTINS["datestr"](interp, [np.array([[days + EPOCH_DATENUM]]), char(fmt or DATE_FORMAT)], 1))


def _start_text(interp, v: Any, fmt: str) -> str:
    """A start date in the stored text form ('' clears it)."""
    if _is_empty_value(v) or (is_text(v) and not text(v).strip()):
        return ""
    dn = _datenums(interp, v)
    if dn.size != 1:
        raise _err("TimeInfo.StartDate must be a single date.", "MATLAB:timeseries:InvalidDate")
    return _datestr(interp, float(dn[0]), fmt)


def _start_num(interp, start_date: str) -> float:
    return float(_datenums(interp, char(start_date))[0])


def _from_days(days, units: str):
    """A difference of day numbers in the given time units (to the microsecond: day numbers carry noise)."""
    return np.round(np.asarray(days, dtype=float) * 86400.0, 6) / UNITS[units]


# -- data layout ----------------------------------------------------------------------------------------

def _check_data(v: Any) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufc":
        raise _err(f"timeseries data must be numeric or logical, not {class_name(v)}.",
                   "MATLAB:timeseries:InvalidData")
    return v


def _align(data: np.ndarray, n: int | None) -> tuple[np.ndarray, bool, int]:
    """(data with an explicit time dimension, time first?, number of samples)."""
    shape = data.shape
    if n is None:
        if data.ndim <= 2:
            if shape[0] == 1 and shape[1] > 1:  # a row vector: one value per time, time last
                return data.reshape(1, 1, shape[1]), False, shape[1]
            return data, True, shape[0]
        return data, False, shape[-1]
    if data.ndim <= 2:
        if shape[0] == n:
            return data, True, n
        if shape[1] == n:
            return data.reshape(shape[0], 1, n), False, n
    elif shape[-1] == n:
        return data, False, n
    raise _err(f"The data ({size_str(shape)}) and the time vector ({n} values) do not fit together: the "
               "number of times must equal the first dimension of a vector or matrix, or the last dimension "
               "of the data.", "MATLAB:timeseries:DataTimeMismatch")


def _time_vector(v: Any) -> np.ndarray:
    a = fnum(v)
    if np.iscomplexobj(a):
        raise _err("The time vector must be real.", "MATLAB:timeseries:InvalidTime")
    if a.size and a.ndim == 2 and min(a.shape) > 1 or a.ndim > 2:
        raise _err("The time vector must be a vector.", "MATLAB:timeseries:InvalidTime")
    t = a.astype(float).reshape(-1, order="F")
    if not np.all(np.isfinite(t)):
        raise _err("The time vector must not contain NaN or Inf values.", "MATLAB:timeseries:InvalidTime")
    return t


def _quality(v: Any, data: np.ndarray, aligned: np.ndarray, n: int) -> np.ndarray | None:
    """Quality codes as int8: a column with one code per sample, or one code per data element.

    ``data`` is the data as given (MATLAB layout), ``aligned`` the same data with its time dimension.
    """
    if v is None or _is_empty_value(v):
        return None
    q = np.asarray(fnum(v))
    if np.iscomplexobj(q) or np.any(q != np.round(q)) or np.any(q < -128) or np.any(q > 127):
        raise _err("Quality codes must be whole numbers from -128 to 127.", "MATLAB:timeseries:InvalidQuality")
    q = q.astype(np.int8)
    if q.shape == data.shape:
        return q.reshape(aligned.shape)
    if q.shape == aligned.shape:
        return q
    if q.ndim == 2 and min(q.shape) == 1 and q.size == n:
        return q.reshape(-1, 1)
    raise _err("The quality codes must be a vector with one code per time or an array the size of the data.",
               "MATLAB:timeseries:InvalidQuality")


def _sort_samples(time: np.ndarray, data: np.ndarray, first: bool, quality):
    order = np.argsort(time, kind="stable")
    if np.array_equal(order, np.arange(time.size)):
        return time, data, quality
    data = data[order] if first else data[..., order]
    if quality is not None:
        if quality.shape == data.shape:
            quality = quality[order] if first else quality[..., order]
        else:
            quality = quality[order]
    return time[order], data, quality


# -- the small value objects -----------------------------------------------------------------------------

def _lines(pairs: list[tuple[str, Any]], head: str, gap: bool = True) -> list[str]:
    width = max(len(k) for k, _ in pairs)
    out = ([f"  {head}", ""] if gap else [f"  {head}"]) if head else []
    for k, v in pairs:
        out.append(f"    {k.rjust(width)}: {v if isinstance(v, str) else summary(v, False)}")
    return out


class _Meta(MObject):
    """Base of the tsdata value objects: named fields, a fixed class name, a properties display."""

    cls = "tsdata.meta"
    FIELDS: tuple = ()

    def class_name(self) -> str:
        return self.cls

    def property_names(self) -> list[str]:
        return list(self.FIELDS)

    def method_names(self) -> list[str]:
        return []

    def field_value(self, interp, name: str) -> Any:
        raise NotImplementedError

    def get_field(self, interp, name: str) -> Any:
        return self.field_value(interp, _resolve(name, self.FIELDS, cls=self.cls))

    def display_lines(self) -> list[str]:
        return _lines([(f, self.field_value(None, f)) for f in self.FIELDS], f"{self.cls} with properties:")

    def object_equals(self, other: Any) -> bool:
        from ..values import values_equal
        return type(other) is type(self) and all(
            values_equal(self.field_value(None, f), other.field_value(None, f)) for f in self.FIELDS)


class TsInterp(_Meta):
    """tsdata.interpolation: the method resample uses ('linear', 'zoh' or a function handle)."""

    cls = "tsdata.interpolation"
    FIELDS = ("Name", "Fhandle")

    def __init__(self, name: str = "linear", fhandle: FunctionHandle | None = None):
        self.name = name
        self.fhandle = fhandle

    def copy(self) -> "TsInterp":
        return TsInterp(self.name, self.fhandle)

    def field_value(self, interp, name: str) -> Any:
        if name == "Name":
            return char(self.name)
        if self.fhandle is not None:
            return self.fhandle
        if interp is None:
            return f"@(t, ts, data) interpolation '{self.name}'"
        kind = "previous" if self.name == "zoh" else "linear"
        return interp.eval_expression(f"@(newTime, time, data) interp1(time, data, newTime, '{kind}')")

    def set_field(self, interp, name: str, value: Any) -> "TsInterp":
        name = _resolve(name, self.FIELDS, cls=self.cls)
        if name == "Name":
            return TsInterp(_interp_name(value), self.fhandle)
        if not isinstance(value, FunctionHandle) and not _is_empty_value(value):
            raise _err("Fhandle must be a function handle.", "MATLAB:tsdata:interpolation:InvalidFhandle")
        return TsInterp(self.name, value if isinstance(value, FunctionHandle) else None)


def _interp_name(v: Any) -> str:
    s = text(v, "interpolation method").lower()
    if s not in ("linear", "zoh"):
        raise _err(f"'{text(v)}' is not an interpolation method of timeseries. Use 'linear', 'zoh' or a function "
                   "handle.", "MATLAB:timeseries:InvalidInterpMethod")
    return s


def _as_interp(v: Any) -> TsInterp:
    if isinstance(v, TsInterp):
        return v.copy()
    if isinstance(v, FunctionHandle):
        from ..display import function_handle_text
        return TsInterp(function_handle_text(v), v)
    return TsInterp(_interp_name(v))


class TsTimeInfo(_Meta):
    """tsdata.timemetadata: units, start date and format of the times, and a summary of them."""

    cls = "tsdata.timemetadata"
    FIELDS = ("Units", "Start", "End", "Increment", "Length", "StartDate", "Format", "UserData")

    def __init__(self, time: np.ndarray, units: str = "seconds", start_date: str = "", fmt: str = "",
                 user_data: Any = None):
        self.time = time
        self.units = units
        self.start_date = start_date
        self.fmt = fmt
        self.user_data = user_data

    def copy(self) -> "TsTimeInfo":
        return TsTimeInfo(self.time.copy(), self.units, self.start_date, self.fmt, self.user_data)

    def increment(self) -> float:
        return _increment(self.time)

    def field_value(self, interp, name: str) -> Any:
        t = self.time
        if name == "Units":
            return char(self.units)
        if name == "Start":
            return np.array([[t[0]]]) if t.size else _empty()
        if name == "End":
            return np.array([[t[-1]]]) if t.size else _empty()
        if name == "Increment":
            return np.array([[self.increment()]])
        if name == "Length":
            return np.array([[float(t.size)]])
        if name == "StartDate":
            return char(self.start_date)
        if name == "Format":
            return char(self.fmt)
        return self.user_data if self.user_data is not None else _empty()

    def set_field(self, interp, name: str, value: Any) -> "TsTimeInfo":
        name = _resolve(name, self.FIELDS, cls=self.cls)
        new = self.copy()
        if name in ("End", "Length"):
            raise MatlabError(f"You cannot set the read-only property '{name}' of {self.cls}.",
                              "MATLAB:class:SetProhibited")
        if name == "Units":
            new.units = _units(value)
        elif name == "Start":
            s = _scalar(value, "start time")
            if new.time.size:
                new.time = new.time - new.time[0] + s
        elif name == "Increment":
            step = _scalar(value, "time increment")
            if not np.isfinite(step) or step <= 0:
                raise _err("The time increment must be a positive finite number.",
                           "MATLAB:timeseries:InvalidIncrement")
            if new.time.size:
                new.time = new.time[0] + step * np.arange(new.time.size)
        elif name == "StartDate":
            if interp is None:
                raise _err("Cannot set the start date here.")
            new.start_date = _start_text(interp, value, self.fmt)
        elif name == "Format":
            new.fmt = "" if _is_empty_value(value) else text(value, "date format")
            if new.start_date and interp is not None:
                new.start_date = _datestr(interp, _start_num(interp, new.start_date), new.fmt)
        else:
            new.user_data = None if (isinstance(value, np.ndarray) and value.size == 0 and not is_char(value)) \
                else value
        return new


def _increment(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    d = np.diff(t)
    scale = max(abs(d[0]), np.max(np.abs(t)), 1e-300)
    if np.all(np.abs(d - d[0]) <= 1e-10 * scale):
        return float(d[0]) if d[0] > 0 else float("nan")
    return float("nan")


class TsDataInfo(_Meta):
    """tsdata.datametadata: the units of the data, its interpolation method and user data."""

    cls = "tsdata.datametadata"
    FIELDS = ("Units", "Interpolation", "UserData")

    def __init__(self, units: str = "", interp: TsInterp | None = None, user_data: Any = None):
        self.units = units
        self.interp = interp or TsInterp()
        self.user_data = user_data

    def copy(self) -> "TsDataInfo":
        return TsDataInfo(self.units, self.interp.copy(), self.user_data)

    def field_value(self, interp, name: str) -> Any:
        if name == "Units":
            return char(self.units)
        if name == "Interpolation":
            return self.interp.copy()
        return self.user_data if self.user_data is not None else _empty()

    def set_field(self, interp, name: str, value: Any) -> "TsDataInfo":
        name = _resolve(name, self.FIELDS, cls=self.cls)
        new = self.copy()
        if name == "Units":
            new.units = "" if _is_empty_value(value) else text(value, "data units")
        elif name == "Interpolation":
            new.interp = _as_interp(value)
        else:
            new.user_data = None if (isinstance(value, np.ndarray) and value.size == 0 and not is_char(value)) \
                else value
        return new


class TsQualityInfo(_Meta):
    """tsdata.qualmetadata: the meaning of each quality code."""

    cls = "tsdata.qualmetadata"
    FIELDS = ("Code", "Description", "UserData")

    def __init__(self, code: np.ndarray | None = None, desc: list[str] | None = None, user_data: Any = None):
        self.code = np.zeros((0, 0)) if code is None else code
        self.desc = list(desc or [])
        self.user_data = user_data

    def copy(self) -> "TsQualityInfo":
        return TsQualityInfo(self.code.copy(), list(self.desc), self.user_data)

    def field_value(self, interp, name: str) -> Any:
        if name == "Code":
            return self.code.reshape(1, -1).copy() if self.code.size else _empty()
        if name == "Description":
            return Cell.from_list([char(d) for d in self.desc]) if self.desc else Cell(np.empty((0, 0), dtype=object))
        return self.user_data if self.user_data is not None else _empty()

    def set_field(self, interp, name: str, value: Any) -> "TsQualityInfo":
        name = _resolve(name, self.FIELDS, cls=self.cls)
        new = self.copy()
        if name == "Code":
            c = fnum(value) if not _is_empty_value(value) else np.zeros((0, 0))
            if c.size and (np.any(c != np.round(c)) or np.any(np.abs(c) > 128)):
                raise _err("Quality codes must be whole numbers from -128 to 127.",
                           "MATLAB:timeseries:InvalidQuality")
            new.code = c.reshape(-1).astype(float)
        elif name == "Description":
            items = _texts(value) if not _is_empty_value(value) else []
            if items is None:
                raise _err("Description must be a cell array of character vectors.")
            new.desc = items
        else:
            new.user_data = value
        return new


class TsEvent(_Meta):
    """tsdata.event: named moments in time (a row of one or more events)."""

    cls = "tsdata.event"
    FIELDS = ("EventData", "Name", "Time", "Units", "StartDate")

    def __init__(self, items: list[dict]):
        self.items = items

    @staticmethod
    def one(name: str, time: float, units: str = "seconds", start_date: str = "", data: Any = None) -> dict:
        return {"Name": name, "Time": time, "Units": units, "StartDate": start_date, "EventData": data}

    def copy(self) -> "TsEvent":
        return TsEvent([dict(e) for e in self.items])

    def matlab_shape(self) -> tuple:
        return (1, len(self.items))

    def field_value(self, interp, name: str) -> Any:
        if len(self.items) != 1:
            raise _err(f"Use one event at a time to read '{name}', for example e(1).{name}.",
                       "MATLAB:tsdata:event:NonScalar")
        e = self.items[0]
        v = e[name]
        if name == "Time":
            return np.array([[v]])
        if name in ("Name", "Units", "StartDate"):
            return char(v)
        return v if v is not None else _empty()

    def set_field(self, interp, name: str, value: Any) -> "TsEvent":
        name = _resolve(name, self.FIELDS, cls=self.cls)
        if len(self.items) != 1:
            raise _err(f"Use one event at a time to set '{name}'.", "MATLAB:tsdata:event:NonScalar")
        new = self.copy()
        e = new.items[0]
        if name == "Name":
            e["Name"] = text(value, "event name")
        elif name == "Time":
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
                if interp is None or not e["StartDate"]:
                    raise _err("An event time given as a date needs the event's StartDate.")
                e["Time"] = float(_from_days(float(_datenums(interp, value)[0]) - _start_num(interp, e["StartDate"]),
                                             e["Units"]))
            else:
                e["Time"] = _scalar(value, "event time")
        elif name == "Units":
            e["Units"] = _units(value)
        elif name == "StartDate":
            e["StartDate"] = _start_text(interp, value, "") if interp is not None else text(value)
        else:
            e["EventData"] = value
        return new

    def index_get(self, interp, args: list) -> Any:
        from .. import indexing as ix
        idx = args[-1] if args else ix.COLON
        if len(args) > 1 and not all(isinstance(a, np.ndarray) and a.size == 1 and float(a.flat[0]) == 1
                                     for a in args[:-1]):
            raise _err("Index exceeds the number of events.", "MATLAB:index:outOfBounds")
        pos, _ = ix._to_positions(idx, len(self.items), for_set=False)
        return TsEvent([dict(self.items[int(p)]) for p in pos])

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        out = []
        for i in items:
            if isinstance(i, np.ndarray) and i.size == 0:
                continue
            if not isinstance(i, TsEvent):
                raise _err("Events can only be concatenated with other tsdata.event objects.")
            out += [dict(e) for e in i.items]
        return TsEvent(out)

    def display_lines(self) -> list[str]:
        if len(self.items) == 1:
            e = self.items[0]
            return _lines([("EventData", e["EventData"] if e["EventData"] is not None else _empty()),
                           ("Name", char(e["Name"])), ("Time", np.array([[e["Time"]]])),
                           ("Units", char(e["Units"])), ("StartDate", char(e["StartDate"]))],
                          f"{self.cls} with properties:")
        return [f"  1×{len(self.items)} {self.cls} array with properties:", ""] + \
            [f"    {f}" for f in self.FIELDS]


def _make_event(interp, args: list) -> TsEvent:
    if len(args) < 2:
        raise _err("tsdata.event needs a name and a time, e.g. tsdata.event('start', 2).", "MATLAB:minrhs")
    name = text(args[0], "event name")
    opts = _name_value(list(args[2:]), ("Units", "StartDate", "EventData"), "tsdata.event")
    units = _units(opts["Units"]) if "Units" in opts else "seconds"
    start = _start_text(interp, opts["StartDate"], "") if "StartDate" in opts else ""
    t = args[1]
    if isinstance(t, np.ndarray) and t.dtype.kind in "biuf":
        time = _scalar(t, "event time")
    else:
        dn = float(_datenums(interp, t)[0])
        if not start:
            start = _datestr(interp, dn)
            time = 0.0
        else:
            time = float(_from_days(dn - _start_num(interp, start), units))
    return TsEvent([TsEvent.one(name, time, units, start, opts.get("EventData"))])


builtin("tsdata.event")(lambda interp, args, nargout: _make_event(interp, args))


@builtin("tsdata.interpolation")
def _tsdata_interpolation(interp, args, nargout):
    if not args:
        return TsInterp()
    return _as_interp(args[0])


# -- the timeseries object ------------------------------------------------------------------------------

class TimeSeries(MObject):
    """A MATLAB timeseries (a value object)."""

    def __init__(self, data: np.ndarray | None = None, time: np.ndarray | None = None, first: bool = True,
                 name: str = "unnamed"):
        self.data = np.zeros((0, 0)) if data is None else data
        self.time = np.zeros(0) if time is None else time
        self.first = first
        self.name = name
        self.quality: np.ndarray | None = None
        self.tinfo = TsTimeInfo(self.time)
        self.dinfo = TsDataInfo()
        self.qinfo = TsQualityInfo()
        self.events: list[dict] = []
        self.user_data: Any = None
        self.treat_nan = True

    # -- copies and shapes -------------------------------------------------------------------------------
    def copy(self) -> "TimeSeries":
        new = TimeSeries(self.data, self.time, self.first, self.name)
        new.quality = self.quality
        new.tinfo = TsTimeInfo(self.time, self.tinfo.units, self.tinfo.start_date, self.tinfo.fmt,
                               self.tinfo.user_data)
        new.dinfo = self.dinfo.copy()
        new.qinfo = self.qinfo.copy()
        new.events = [dict(e) for e in self.events]
        new.user_data = self.user_data
        new.treat_nan = self.treat_nan
        return new

    def with_samples(self, time: np.ndarray, samples: np.ndarray, quality: np.ndarray | None = None) -> "TimeSeries":
        """A copy holding new samples (time along axis 0 of ``samples`` and of ``quality``)."""
        new = self.copy()
        new.time = np.asarray(time, dtype=float).reshape(-1)
        new.tinfo.time = new.time
        new.data = self._from_samples(samples)
        new.quality = new.quality_from_samples(quality)
        return new

    def _from_samples(self, s: np.ndarray) -> np.ndarray:
        if self.first:
            return s.reshape(s.shape[0], -1) if s.ndim != 2 else s
        return np.moveaxis(s, 0, -1)

    @property
    def n(self) -> int:
        return int(self.time.size)

    def samples(self) -> np.ndarray:
        """The data with time along axis 0."""
        d = self.data
        if self.first:
            return d if d.ndim == 2 else d.reshape(d.shape[0], -1)
        return np.moveaxis(d, -1, 0)

    def sample_shape(self) -> tuple:
        if self.first:
            return (1, self.data.shape[1] if self.data.ndim == 2 else 1)
        return tuple(self.data.shape[:-1])

    def quality_samples(self) -> np.ndarray | None:
        """Quality codes with time along axis 0: (n,) for one code per sample, else one per element."""
        q = self.quality
        if q is None:
            return None
        if q.shape == self.data.shape:
            return q if self.first else np.moveaxis(q, -1, 0)
        return q.reshape(-1)

    def quality_from_samples(self, q: np.ndarray | None) -> np.ndarray | None:
        if q is None:
            return None
        if q.ndim == 1:
            return q.reshape(-1, 1)
        return self._from_samples(q)

    def data_value(self) -> np.ndarray:
        return as_2d(self.data.copy())

    def time_value(self) -> np.ndarray:
        return self.time.reshape(-1, 1).copy()

    # -- identity --------------------------------------------------------------------------------------------
    def class_name(self) -> str:
        return "timeseries"

    def property_names(self) -> list[str]:
        return list(PROPS)

    def method_names(self) -> list[str]:
        return sorted(set(_METHODS) | {"timeseries"}, key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in _METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if args and not isinstance(args[0], TimeSeries) and name not in ("plot", "append") and name in BUILTINS:
            # a timeseries passed further along, as in set(h, 'UserData', ts): the ordinary function
            return interp.call_builtin(name, BUILTINS[name], list(args), nargout)
        res = _METHODS[name](interp, args, nargout)
        if res is None:
            return []
        return list(res) if isinstance(res, tuple) else [res]

    def object_equals(self, other: Any) -> bool:
        from ..values import values_equal
        if not isinstance(other, TimeSeries):
            return False
        def same(a, b):
            return values_equal(a if a is not None else _empty(), b if b is not None else _empty())

        def same_events(a, b):
            keys = ("Name", "Time", "Units", "StartDate")
            return len(a) == len(b) and all(all(x[k] == y[k] for k in keys) and same(x["EventData"], y["EventData"])
                                            for x, y in zip(a, b))

        return (self.name == other.name and self.first == other.first and self.data.shape == other.data.shape
                and np.array_equal(self.data, other.data, equal_nan=self.data.dtype.kind in "fc")
                and np.array_equal(self.time, other.time) and self.tinfo.units == other.tinfo.units
                and self.tinfo.start_date == other.tinfo.start_date and self.tinfo.fmt == other.tinfo.fmt
                and same(self.quality, other.quality) and same(self.user_data, other.user_data)
                and self.treat_nan == other.treat_nan
                and self.dinfo.units == other.dinfo.units and self.dinfo.interp.name == other.dinfo.interp.name
                and same_events(self.events, other.events))

    # -- properties ------------------------------------------------------------------------------------------
    def get_field(self, interp, name: str) -> Any:
        name = _resolve(name, PROPS)
        if name == "Data":
            return self.data_value()
        if name == "Time":
            return self.time_value()
        if name == "Name":
            return char(self.name)
        if name == "Length":
            return np.array([[float(self.n)]])
        if name == "IsTimeFirst":
            return bool_scalar(self.first)
        if name == "TreatNaNasMissing":
            return bool_scalar(self.treat_nan)
        if name == "Quality":
            return as_2d(self.quality.copy()) if self.quality is not None else _empty()
        if name == "TimeInfo":
            return TsTimeInfo(self.time.copy(), self.tinfo.units, self.tinfo.start_date, self.tinfo.fmt,
                              self.tinfo.user_data)
        if name == "DataInfo":
            return self.dinfo.copy()
        if name == "QualityInfo":
            return self.qinfo.copy()
        if name == "Events":
            return TsEvent([dict(e) for e in self.events]) if self.events else _empty()
        return self.user_data if self.user_data is not None else _empty()

    def set_field(self, interp, name: str, value: Any) -> "TimeSeries":
        name = _resolve(name, PROPS)
        if name in READ_ONLY:
            raise MatlabError(f"You cannot set the read-only property '{name}' of timeseries.",
                              "MATLAB:class:SetProhibited")
        new = self.copy()
        if name == "Name":
            new.name = "" if _is_empty_value(value) else text(value, "name")
        elif name == "Data":
            new._set_data(value)
        elif name == "Time":
            new._set_time(interp, value)
        elif name == "Quality":
            new.quality = _quality(value, as_2d(new.data), new.data, new.n)
        elif name == "TimeInfo":
            if not isinstance(value, TsTimeInfo):
                raise _err("TimeInfo must be a tsdata.timemetadata object.")
            if value.time.size != new.n:
                raise _err(f"The time information describes {value.time.size} times, but the timeseries has "
                           f"{new.n} samples.", "MATLAB:timeseries:DataTimeMismatch")
            new.tinfo = value.copy()
            new.time = new.tinfo.time
        elif name == "DataInfo":
            if not isinstance(value, TsDataInfo):
                raise _err("DataInfo must be a tsdata.datametadata object.")
            new.dinfo = value.copy()
        elif name == "QualityInfo":
            if not isinstance(value, TsQualityInfo):
                raise _err("QualityInfo must be a tsdata.qualmetadata object.")
            new.qinfo = value.copy()
        elif name == "Events":
            if _is_empty_value(value):
                new.events = []
            elif isinstance(value, TsEvent):
                new.events = [dict(e) for e in value.items]
            else:
                raise _err("Events must be tsdata.event objects.")
        elif name == "TreatNaNasMissing":
            new.treat_nan = _flag(value, "TreatNaNasMissing value")
        else:
            new.user_data = None if (isinstance(value, np.ndarray) and value.size == 0 and not is_char(value)) \
                else value
        return new

    def _set_data(self, value: Any) -> None:
        d = _check_data(value)
        if self.n == 0 and d.size:
            d, first, n = _align(d, None)
            self.time = np.arange(float(n))
            self.tinfo.time = self.time
        else:
            d, first, _ = _align(d, self.n)
        if self.quality is not None and self.quality.shape != d.shape and self.quality.size != self.n:
            self.quality = None
        self.data, self.first = d, first

    def _set_time(self, interp, value: Any) -> None:
        start = self.tinfo.start_date
        units = self.tinfo.units
        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            t = _time_vector(value)
        else:
            dn = _datenums(interp, value)
            if start:
                base = _start_num(interp, start)
            else:
                base = float(np.min(dn)) if dn.size else 0.0
                start = _datestr(interp, base, self.tinfo.fmt)
                units = "days"
            t = _from_days(dn - base, units)
        if t.size != self.n and not (self.n == 0 and self.data.size == 0 and t.size == 0):
            raise _err(f"The new time vector has {t.size} values, but the timeseries has {self.n} samples; "
                       "change Time and Data together with timeseries(data, time) or set Data first.",
                       "MATLAB:timeseries:DataTimeMismatch")
        if np.any(np.diff(t) < 0):
            raise _err("The time vector must be in increasing order.", "MATLAB:timeseries:InvalidTime")
        self.time = t
        self.tinfo.time = t
        self.tinfo.start_date = start
        self.tinfo.units = units

    # -- display ---------------------------------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        pairs = [("Name", char(self.name)), ("Time", self.time_value()), ("TimeInfo", self.get_field(None, "TimeInfo")),
                 ("Data", self.data_value()), ("DataInfo", self.dinfo)]
        if self.events:
            pairs.append(("Events", self.get_field(None, "Events")))
        if self.quality is not None:
            pairs.append(("Quality", self.get_field(None, "Quality")))
        return ["  timeseries", ""] + _lines(pairs, "Common Properties:", gap=False)

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"

    def summary_text(self, in_cell: bool) -> str:
        return "1×1 timeseries" if in_cell else "[1×1 timeseries]"

    # -- indexing and operators ------------------------------------------------------------------------------
    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        parts = [i for i in items if not (isinstance(i, np.ndarray) and i.size == 0)]
        if len(parts) == 1 and isinstance(parts[0], TimeSeries):
            return parts[0]
        if axis == 0 and all(isinstance(p, TimeSeries) for p in parts):
            return _append(None, parts)  # [ts1; ts2] joins them in time, like append
        raise _err("Arrays of timeseries objects are not supported; keep several in a cell array, or use "
                   "append (or [ts1; ts2]) to join them in time.", "MATLAB:timeseries:NoArrays")

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("+", "-", ".*", "./", ".\\", ".^", "*", "/", "\\"):
            return NotImplemented
        return _arith(op, left, right)

    def unary_op(self, interp, op: str) -> Any:
        if op in ("-", "+"):
            return self.with_samples(self.time, ops.unary(op, self.samples()), self.quality_samples())
        return NotImplemented


# -- construction ---------------------------------------------------------------------------------------

def build(interp, args: list) -> TimeSeries:
    rest = list(args)
    opts: dict = {}
    known = ("Name", "IsTimeFirst")
    while len(rest) >= 2 and is_text(rest[-2]) and match_option(text(rest[-2]), known, partial=False):
        opts.setdefault(match_option(text(rest[-2]), known, partial=False), rest[-1])
        del rest[-2:]
    name = text(opts["Name"], "name") if "Name" in opts else None
    if not rest:
        return TimeSeries(name=name if name is not None else "unnamed")
    if len(rest) == 1 and is_text(rest[0]):  # timeseries(name): an empty timeseries with that name
        return TimeSeries(name=text(rest[0]))
    if len(rest) >= 2 and is_text(rest[-1]) and not (len(rest) == 2 and _looks_like_date(interp, rest[-1])):
        name = text(rest.pop())  # the name as the last positional argument
    if len(rest) > 3:
        raise _err("Too many input arguments for timeseries.", "MATLAB:TooManyInputs")
    data = _check_data(rest[0])
    time_arg = rest[1] if len(rest) > 1 else None
    quality_arg = rest[2] if len(rest) > 2 else None
    start, units = "", "seconds"
    if time_arg is None or (isinstance(time_arg, np.ndarray) and time_arg.size == 0 and data.size):
        aligned, first, n = _align(data, None)
        time = np.arange(float(n))
    else:
        if isinstance(time_arg, np.ndarray) and time_arg.dtype.kind in "biuf":
            time = _time_vector(time_arg)
        else:
            dn = _datenums(interp, time_arg)
            base = float(np.min(dn)) if dn.size else 0.0
            start, units = _datestr(interp, base), "days"
            time = _from_days(dn - base, "days")
        aligned, first, n = _align(data, time.size)
    if "IsTimeFirst" in opts and _flag(opts["IsTimeFirst"], "IsTimeFirst value") != first:
        raise _err("IsTimeFirst does not match the layout of the data: with one or two dimensions time runs "
                   "along the rows, with more along the last dimension.", "MATLAB:timeseries:DataTimeMismatch")
    quality = _quality(quality_arg, data, aligned, n)
    time, aligned, quality = _sort_samples(time, aligned, first, quality)
    ts = TimeSeries(aligned, time, first, name if name is not None else "unnamed")
    ts.quality = quality
    ts.tinfo.units = units
    ts.tinfo.start_date = start
    return ts


def _looks_like_date(interp, v: Any) -> bool:
    s = text(v)
    if not any(c.isdigit() for c in s):
        return False
    try:
        _datenums(interp, v)
        return True
    except MatlabError:
        return False


@builtin("timeseries")
def _timeseries(interp, args, nargout):
    return build(interp, args)


def _need_ts(v: Any, fname: str) -> TimeSeries:
    if not isinstance(v, TimeSeries):
        raise MatlabError(f"Undefined function '{fname}' for input arguments of type '{class_name(v)}'.",
                          "MATLAB:UndefinedFunction")
    return v


# -- times given as numbers or dates --------------------------------------------------------------------

def _rel_times(interp, ts: TimeSeries, v: Any, what: str = "time") -> np.ndarray:
    """Times in the units of ts, relative to its start date: numbers, date texts or datetime values."""
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        a = fnum(v)
        if np.iscomplexobj(a):
            raise _err(f"The {what} must be real.")
        return a.astype(float).reshape(-1, order="F")
    if not ts.tinfo.start_date:
        raise _err(f"A {what} given as a date needs TimeInfo.StartDate to be set.",
                   "MATLAB:timeseries:NoStartDate")
    dn = _datenums(interp, v)
    return _from_days(dn - _start_num(interp, ts.tinfo.start_date), ts.tinfo.units)


def _abs_seconds(interp, ts: TimeSeries, ref: float | None) -> np.ndarray:
    """Times of ts in seconds from the day number ``ref`` (or plain seconds without dates)."""
    secs = ts.time * UNITS[ts.tinfo.units]
    if ts.tinfo.start_date and ref is not None:
        secs = secs + (_start_num(interp, ts.tinfo.start_date) - ref) * 86400.0
    return secs


# -- interpolation -----------------------------------------------------------------------------------------

def _interp_samples(interp, ts: TimeSeries, tq: np.ndarray, method: TsInterp) -> tuple[np.ndarray, bool]:
    """Samples of ts at the times tq: (array with time along axis 0, whether some fell outside)."""
    t = ts.time
    S = ts.samples()
    n = t.size
    rest = S.shape[1:]
    if method.fhandle is not None:
        res = interp.call_function_value(method.fhandle, [tq.reshape(-1, 1), t.reshape(-1, 1),
                                                          as_2d(S.reshape(n, -1))], 1)
        if not res:
            raise _err("The interpolation function returned nothing.", "MATLAB:timeseries:InvalidInterpMethod")
        out = fnum(res[0])
        return out.reshape((tq.size,) + rest), False
    flat = S.reshape(n, -1)
    out_dtype = complex if np.iscomplexobj(flat) else float
    out = np.full((tq.size, flat.shape[1]), np.nan, dtype=out_dtype)
    if n == 0:
        return out.reshape((tq.size,) + rest), bool(tq.size)
    outside = (tq < t[0]) | (tq > t[-1])
    inside = ~outside & ~np.isnan(tq)
    if method.name == "zoh":
        idx = np.searchsorted(t, tq[inside], side="right") - 1
        out[inside] = flat[np.clip(idx, 0, n - 1)]
    elif n == 1:
        out[inside] = flat[0]
    else:
        # repeated times: keep the last sample at each time
        keep = np.append(np.diff(t) > 0, True)
        tt, ff = t[keep], flat[keep].astype(out_dtype)
        if tt.size == 1:
            out[inside] = ff[0]
        else:
            q = tq[inside]
            i = np.clip(np.searchsorted(tt, q, side="right") - 1, 0, tt.size - 2)
            w = ((q - tt[i]) / (tt[i + 1] - tt[i]))[:, None]
            exact = q == tt[i]
            lo, hi = ff[i], ff[i + 1]
            with np.errstate(invalid="ignore"):
                vals = lo + w * (hi - lo)
            vals[exact] = lo[exact]
            vals[q == tt[i + 1]] = hi[q == tt[i + 1]]
            out[inside] = vals
    return out.reshape((tq.size,) + rest), bool(np.any(outside))


def _noextrap_warning(interp) -> None:
    interp.warn("Some new times lie outside the time range of the timeseries; the data there is NaN "
                "(timeseries values are not extrapolated).", "interpolation:interpolation:noextrap")


def _quality_at(ts: TimeSeries, tq: np.ndarray, code: Any) -> np.ndarray | None:
    if code is not None and not _is_empty_value(code):
        c = fnum(code)
        if c.size != 1:
            raise _err("The quality code must be a single number.", "MATLAB:timeseries:InvalidQuality")
        return np.full(tq.size, int(c.reshape(-1)[0]), dtype=np.int8)
    q = ts.quality_samples()
    if q is None or ts.n == 0:
        return None
    idx = np.clip(np.searchsorted(ts.time, tq, side="right") - 1, 0, ts.n - 1)
    return q[idx]


def _resample_to(interp, ts: TimeSeries, tq: np.ndarray, method: TsInterp | None = None, code: Any = None,
                 warn: bool = True) -> TimeSeries:
    method = method or ts.dinfo.interp
    tq = np.sort(np.asarray(tq, dtype=float).reshape(-1), kind="stable")
    S, outside = _interp_samples(interp, ts, tq, method)
    if outside and warn:
        _noextrap_warning(interp)
    return ts.with_samples(tq, S, _quality_at(ts, tq, code))


# -- methods ----------------------------------------------------------------------------------------------

_METHODS: dict = {}


def _method(*names: str):
    def deco(fn):
        for n in names:
            _METHODS[n] = fn
        return fn
    return deco


def _ts_and_rest(args: list, fname: str) -> tuple[TimeSeries, list]:
    if not args:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    return _need_ts(args[0], fname), list(args[1:])


@_method("resample")
def _m_resample(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "resample")
    if not rest:
        raise MatlabError("resample needs the new times, resample(ts, timevec).", "MATLAB:minrhs")
    tq = _rel_times(interp, ts, rest[0], "new time")
    method = _as_interp(rest[1]) if len(rest) > 1 and not _is_empty_value(rest[1]) else None
    code = rest[2] if len(rest) > 2 else None
    if len(rest) > 3:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
    return _resample_to(interp, ts, tq, method, code)


@_method("get")
def _m_get(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "get")
    if not rest:
        s = Struct.scalar({p: ts.get_field(interp, p) for p in PROPS})
        if nargout:
            return s
        from ..display import display_lines
        interp.out("\n".join(display_lines(s)) + "\n")
        return None
    if isinstance(rest[0], Cell):
        return Cell.from_list([ts.get_field(interp, to_str(p)) for p in rest[0].values_f()])
    return ts.get_field(interp, text(rest[0], "property name"))


@_method("set")
def _m_set(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "set")
    if not rest:
        if nargout:
            return Cell.from_list([char(p) for p in PROPS if p not in READ_ONLY], column=True)
        interp.out("".join(f"    {p}\n" for p in PROPS if p not in READ_ONLY))
        return None
    if len(rest) % 2:
        raise _err("set needs property names and values in pairs, set(ts, 'Name', value).",
                   "MATLAB:class:BadParamValuePairs")
    new = ts
    for k in range(0, len(rest), 2):
        new = new.set_field(interp, text(rest[k], "property name"), rest[k + 1])
    if nargout:
        return new
    names = interp.state.get("call_arg_names") or []
    if names and names[0] and interp.frames:
        interp.frames[-1].set(names[0], new)  # the variable passed in takes the new values
    return None


@_method("getdatasamples")
def _m_getdatasamples(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "getdatasamples")
    if len(rest) != 1:
        raise MatlabError("getdatasamples needs the sample indices, getdatasamples(ts, ind).", "MATLAB:minrhs")
    pos = _positions(rest[0], ts.n)
    return as_2d(ts._from_samples(ts.samples()[pos]))


def _positions(v: Any, n: int) -> np.ndarray:
    from .. import indexing as ix
    pos, _ = ix._to_positions(v, n, for_set=False)
    return np.asarray(pos, dtype=int).reshape(-1)


def _subset(ts: TimeSeries, pos: np.ndarray) -> TimeSeries:
    q = ts.quality_samples()
    return ts.with_samples(ts.time[pos], ts.samples()[pos], None if q is None else q[pos])


@_method("getsamples")
def _m_getsamples(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "getsamples")
    if len(rest) != 1:
        raise MatlabError("getsamples needs the sample indices, getsamples(ts, ind).", "MATLAB:minrhs")
    return _subset(ts, _positions(rest[0], ts.n))


def _close(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(a - b) <= TOL * max(1.0, abs(b))


@_method("getsampleusingtime")
def _m_getsampleusingtime(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "getsampleusingtime")
    if len(rest) not in (1, 2):
        raise MatlabError("getsampleusingtime needs a time, or a start and an end time.", "MATLAB:minrhs")
    t0 = _rel_times(interp, ts, rest[0])
    if t0.size != 1:
        raise _err("The time must be a single value.")
    if len(rest) == 1:
        mask = _close(ts.time, float(t0[0]))
    else:
        t1 = _rel_times(interp, ts, rest[1])
        if t1.size != 1:
            raise _err("The end time must be a single value.")
        lo, hi = float(t0[0]), float(t1[0])
        mask = ((ts.time >= lo) | _close(ts.time, lo)) & ((ts.time <= hi) | _close(ts.time, hi))
    return _subset(ts, np.nonzero(mask)[0])


@_method("getdatasamplesize")
def _m_getdatasamplesize(interp, args, nargout):
    ts, _ = _ts_and_rest(args, "getdatasamplesize")
    return np.array([[float(d) for d in ts.sample_shape()]])


@_method("getinterpmethod")
def _m_getinterpmethod(interp, args, nargout):
    ts, _ = _ts_and_rest(args, "getinterpmethod")
    return char(ts.dinfo.interp.name)


@_method("setinterpmethod")
def _m_setinterpmethod(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "setinterpmethod")
    if not rest:
        raise MatlabError("setinterpmethod needs a method: 'linear', 'zoh' or a function handle.", "MATLAB:minrhs")
    new = ts.copy()
    new.dinfo.interp = _as_interp(rest[0])
    return new


@_method("isempty")
def _m_isempty(interp, args, nargout):
    return bool_scalar(_need_ts(args[0], "isempty").n == 0)


@_method("length")
def _m_length(interp, args, nargout):
    return np.array([[float(_need_ts(args[0], "length").n)]])


@_method("size")
def _m_size(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "size")
    dims = [float(ts.n), 1.0]
    if rest:
        k = int(_scalar(rest[0], "dimension"))
        if k < 1:
            raise MatlabError("The dimension must be a positive whole number.",
                              "MATLAB:getdimarg:dimensionMustBePositiveInteger")
        return np.array([[dims[k - 1] if k <= 2 else 1.0]])
    if nargout <= 1:
        return np.array([dims])
    return tuple(np.array([[d]]) for d in dims + [1.0] * (nargout - 2))


@_method("getabstime")
def _m_getabstime(interp, args, nargout):
    ts, _ = _ts_and_rest(args, "getabstime")
    if not ts.tinfo.start_date:
        raise _err("getabstime needs TimeInfo.StartDate to be set.", "MATLAB:timeseries:NoStartDate")
    base = _start_num(interp, ts.tinfo.start_date)
    days = base + ts.time * UNITS[ts.tinfo.units] / 86400.0
    return Cell.from_list([char(_datestr(interp, float(d), ts.tinfo.fmt)) for d in days], column=True)


@_method("setabstime")
def _m_setabstime(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "setabstime")
    if not rest:
        raise MatlabError("setabstime needs the times as dates.", "MATLAB:minrhs")
    new = ts.copy()
    if len(rest) > 1:
        new.tinfo.fmt = text(rest[1], "date format")
    dn = _datenums(interp, rest[0])
    if dn.size != ts.n:
        raise _err(f"setabstime needs {ts.n} dates, one per sample.", "MATLAB:timeseries:DataTimeMismatch")
    if np.any(np.diff(dn) < 0):
        raise _err("The dates must be in increasing order.", "MATLAB:timeseries:InvalidTime")
    base = float(dn[0]) if dn.size else 0.0
    new.tinfo.start_date = _datestr(interp, base, new.tinfo.fmt)
    new.time = _from_days(dn - base, new.tinfo.units)
    new.tinfo.time = new.time
    return new


@_method("setuniformtime")
def _m_setuniformtime(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "setuniformtime")
    opts = _name_value(rest, ("StartTime", "Interval", "EndTime"), "setuniformtime")
    n = ts.n
    start = _scalar(opts["StartTime"], "start time") if "StartTime" in opts else None
    step = _scalar(opts["Interval"], "interval") if "Interval" in opts else None
    end = _scalar(opts["EndTime"], "end time") if "EndTime" in opts else None
    if start is None and end is not None and step is not None:
        start = end - step * (n - 1)
    if start is None:
        start = float(ts.time[0]) if n else 0.0
    if step is None:
        step = (end - start) / (n - 1) if end is not None and n > 1 else (_increment(ts.time) if n > 1 else 1.0)
        if not np.isfinite(step):
            step = 1.0
    new = ts.copy()
    new.time = start + step * np.arange(n)
    new.tinfo.time = new.time
    return new


@_method("getqualitydesc")
def _m_getqualitydesc(interp, args, nargout):
    ts, _ = _ts_and_rest(args, "getqualitydesc")
    q = ts.quality_samples()
    if q is None:
        return Cell(np.empty((0, 0), dtype=object))
    codes = list(ts.qinfo.code.reshape(-1))
    flat = q.reshape(q.shape[0], -1)[:, 0] if q.ndim > 1 else q
    out = []
    for c in flat:
        k = next((i for i, x in enumerate(codes) if x == c), None)
        out.append(char(ts.qinfo.desc[k] if k is not None and k < len(ts.qinfo.desc) else ""))
    return Cell.from_list(out, column=True)


# -- combining ------------------------------------------------------------------------------------------

def _in_units(interp, ts: TimeSeries, units: str, start_date: str) -> np.ndarray:
    """Times of ts expressed in ``units`` from ``start_date`` (or plain, without dates)."""
    ref = _start_num(interp, start_date) if start_date and ts.tinfo.start_date else None
    return _abs_seconds(interp, ts, ref) / UNITS[units]


def _append(interp, parts: list) -> TimeSeries:
    first = parts[0]
    times, samples, qual = [first.time], [first.samples()], [first.quality_samples()]
    shape = first.samples().shape[1:]
    events = [dict(e) for e in first.events]
    for p in parts[1:]:
        p = _need_ts(p, "append")
        if p.n and p.samples().shape[1:] != shape and first.n:
            raise _err("The timeseries must have samples of the same size to be appended.",
                       "MATLAB:timeseries:append:SampleSizeMismatch")
        t = p.time if (p.tinfo.units == first.tinfo.units and p.tinfo.start_date == first.tinfo.start_date) else \
            _in_units(interp, p, first.tinfo.units, first.tinfo.start_date)
        last = np.concatenate(times)[-1] if sum(x.size for x in times) else -np.inf
        if t.size and t[0] < last - TOL * max(1.0, abs(last)):
            raise _err("The time vectors overlap: each timeseries must start after the previous one ends.",
                       "MATLAB:timeseries:append:OverlappingTimes")
        times.append(t)
        samples.append(p.samples() if p.n else np.zeros((0,) + shape))
        qual.append(p.quality_samples())
        events += [dict(e) for e in p.events]
    t_all = np.concatenate(times)
    base = first
    if not first.n:
        donor = next((p for p in parts if p.n), None)
        if donor is not None:
            shape = donor.samples().shape[1:]
            samples = [s if s.shape[0] else np.zeros((0,) + shape) for s in samples]
            base = first.copy()
            base.first = donor.first
    S = np.concatenate(samples, axis=0)
    q = None
    if all(x is not None for x in qual) and len({x.ndim for x in qual}) == 1:
        q = np.concatenate(qual, axis=0)
    new = base.with_samples(t_all, S, q)
    new.events = events
    return new


@_method("append")
def _m_append(interp, args, nargout):
    parts = [a for a in args]
    if not parts:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    if not all(isinstance(p, TimeSeries) for p in parts):
        bad = next(p for p in parts if not isinstance(p, TimeSeries))
        raise _err(f"append joins timeseries objects; one input is of class '{class_name(bad)}'.")
    return _append(interp, parts)


def _sample_value(v: Any) -> np.ndarray:
    return _check_data(v)


@_method("addsample")
def _m_addsample(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "addsample")
    if len(rest) == 1 and isinstance(rest[0], Struct):
        s = rest[0]
        rest = []
        for f in s.fields:
            rest += [char(f), s.get_field(0, f)]
    opts = _name_value(rest, ("Data", "Time", "Quality", "OverwriteFlag"), "addsample")
    if "Data" not in opts or "Time" not in opts:
        raise _err("addsample needs both 'Data' and 'Time'.", "MATLAB:timeseries:addsample:MissingInput")
    t_new = _rel_times(interp, ts, opts["Time"])
    d = _sample_value(opts["Data"])
    k = t_new.size
    if ts.n == 0 and ts.data.size == 0:
        aligned, first, _ = _align(d, k)
        S_new = np.moveaxis(aligned, -1, 0) if not first else aligned
        base = TimeSeries(aligned, t_new, first, ts.name)
        template = ts.copy()
        template.data, template.first = base.data, base.first
        S_old = np.zeros((0,) + S_new.shape[1:])
    else:
        template = ts
        shape = ts.samples().shape[1:]
        if d.size != k * int(np.prod(shape)):
            raise _err(f"The new data must hold {k} sample(s) of size {size_str(ts.sample_shape())}.",
                       "MATLAB:timeseries:addsample:SampleSizeMismatch")
        if ts.first:
            S_new = d.reshape((k,) + shape) if d.shape[0] == k else d.reshape((k,) + shape, order="F")
        else:
            S_new = np.moveaxis(d.reshape(shape + (k,), order="F"), -1, 0)
        S_old = ts.samples()
    overwrite = _flag(opts["OverwriteFlag"], "OverwriteFlag") if "OverwriteFlag" in opts else False
    q_old = template.quality_samples() if template.n else None
    q_new = None
    if "Quality" in opts and not _is_empty_value(opts["Quality"]):
        qa = fnum(opts["Quality"]).astype(np.int8)
        q_new = qa.reshape(-1) if qa.size == k else qa.reshape(S_new.shape, order="F")
    t_old = template.time
    keep = np.ones(t_old.size, dtype=bool)
    if overwrite:
        for t in t_new:
            keep &= ~_close(t_old, float(t))
    t_all = np.concatenate([t_old[keep], t_new])
    S_all = np.concatenate([S_old[keep].astype(np.result_type(S_old, S_new)), S_new], axis=0)
    order = np.argsort(t_all, kind="stable")
    qa = None
    if q_old is not None or q_new is not None:
        per_elem = (q_old is not None and q_old.ndim > 1) or (q_new is not None and q_new.ndim > 1)
        def full(q, count):
            if q is None:
                return np.zeros((count,) + (S_all.shape[1:] if per_elem else ()), np.int8)
            if per_elem and q.ndim == 1:
                return np.broadcast_to(q.reshape((-1,) + (1,) * (S_all.ndim - 1)), (count,) + S_all.shape[1:])
            return q
        qa = np.concatenate([full(None if q_old is None else q_old[keep], int(keep.sum())), full(q_new, k)],
                            axis=0)[order]
    return template.with_samples(t_all[order], S_all[order], qa)


@_method("delsample")
def _m_delsample(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "delsample")
    if len(rest) != 2 or not is_text(rest[0]):
        raise _err("delsample needs 'Index', indices or 'Value', times.", "MATLAB:timeseries:delsample:BadInput")
    how = match_option(text(rest[0]), ("Index", "Value"))
    if how is None:
        raise _err(f"delsample: unknown option '{text(rest[0])}'. Use 'Index' or 'Value'.",
                   "MATLAB:timeseries:delsample:BadInput")
    drop = np.zeros(ts.n, dtype=bool)
    if how == "Index":
        drop[_positions(rest[1], ts.n)] = True
    else:
        for t in _rel_times(interp, ts, rest[1]):
            drop |= _close(ts.time, float(t))
    return _subset(ts, np.nonzero(~drop)[0])


def _union_times(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    t = np.sort(np.concatenate([a, b]))
    if t.size == 0:
        return t
    keep = np.append(True, np.diff(t) > tol)
    return t[keep]


@_method("synchronize")
def _m_synchronize(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("synchronize needs two timeseries and a method: 'Union', 'Intersection' or 'Uniform'.",
                          "MATLAB:minrhs")
    ts1, ts2 = _need_ts(args[0], "synchronize"), _need_ts(args[1], "synchronize")
    how = match_option(text(args[2], "synchronize method"), ("Union", "Intersection", "Uniform"))
    if how is None:
        raise _err(f"synchronize: unknown method '{text(args[2])}'. Use 'Union', 'Intersection' or 'Uniform'.",
                   "MATLAB:timeseries:synchronize:BadMethod")
    opts = _name_value(list(args[3:]), ("Interval", "InterpMethod", "QualityCode", "KeepOriginalTimes",
                                        "tolerance"), "synchronize")
    tol = _scalar(opts["tolerance"], "tolerance") if "tolerance" in opts else TOL
    method = _as_interp(opts["InterpMethod"]) if "InterpMethod" in opts else None
    code = opts.get("QualityCode")
    keep_own = _flag(opts["KeepOriginalTimes"], "KeepOriginalTimes") if "KeepOriginalTimes" in opts else False
    # a common clock: the finer of the two units, counted from the earlier start date
    units = min((ts1.tinfo.units, ts2.tinfo.units), key=lambda u: UNITS[u])
    dated = bool(ts1.tinfo.start_date and ts2.tinfo.start_date)
    if dated:
        s1, s2 = _start_num(interp, ts1.tinfo.start_date), _start_num(interp, ts2.tinfo.start_date)
        ref = min(s1, s2)
        start = ts1.tinfo.start_date if s1 <= s2 else ts2.tinfo.start_date
    else:
        ref, start = None, ""
    t1 = _abs_seconds(interp, ts1, ref) / UNITS[units]
    t2 = _abs_seconds(interp, ts2, ref) / UNITS[units]
    if not t1.size or not t2.size:
        raise _err("synchronize needs two nonempty timeseries.", "MATLAB:timeseries:synchronize:Empty")
    lo, hi = max(t1[0], t2[0]), min(t1[-1], t2[-1])
    if lo > hi + tol:
        raise _err("The time ranges of the two timeseries do not overlap, so they cannot be synchronized.",
                   "MATLAB:timeseries:synchronize:NoOverlap")
    if how == "Union":
        inside = lambda t: t[(t >= lo - tol) & (t <= hi + tol)]  # noqa: E731
        common = _union_times(inside(t1), inside(t2), tol)
    elif how == "Intersection":
        common = np.array([x for x in t1 if np.any(np.abs(t2 - x) <= tol)])
    else:
        if "Interval" not in opts:
            raise _err("synchronize(..., 'Uniform') needs 'Interval', step.",
                       "MATLAB:timeseries:synchronize:NoInterval")
        step = _scalar(opts["Interval"], "interval")
        if not np.isfinite(step) or step <= 0:
            raise _err("The interval must be a positive number.", "MATLAB:timeseries:synchronize:BadInterval")
        common = lo + step * np.arange(int(np.floor((hi - lo) / step + 1e-9)) + 1)
    outs = []
    for ts, t in ((ts1, t1), (ts2, t2)):
        moved = ts.copy()
        moved.time = t
        moved.tinfo.time = t
        moved.tinfo.units = units
        moved.tinfo.start_date = start if dated else ""
        r = _resample_to(interp, moved, common, method, code, warn=False)
        if dated and keep_own:
            shift = _from_days(_start_num(interp, ts.tinfo.start_date) - ref, units)
            r.time = r.time - shift
            r.tinfo.time = r.time
            r.tinfo.start_date = ts.tinfo.start_date
        elif not dated:
            r.tinfo.start_date = ts.tinfo.start_date
        outs.append(r)
    return tuple(outs)


# -- events ---------------------------------------------------------------------------------------------

@_method("addevent")
def _m_addevent(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "addevent")
    if not rest:
        raise MatlabError("addevent needs an event, or a name and a time.", "MATLAB:minrhs")
    if isinstance(rest[0], TsEvent):
        ev = [dict(e) for e in rest[0].items]
    elif isinstance(rest[0], Cell) or len(rest) == 2:
        names = _texts(rest[0]) or []
        tv = rest[1] if len(rest) > 1 else None
        if tv is None:
            raise MatlabError("addevent needs a time for each event name.", "MATLAB:minrhs")
        if isinstance(tv, np.ndarray) and tv.dtype.kind in "biuf":
            times = list(fnum(tv).reshape(-1))
            ev = [TsEvent.one(nm, float(t), ts.tinfo.units, ts.tinfo.start_date) for nm, t in zip(names, times)]
        else:
            rel = _rel_times(interp, ts, tv, "event time")
            ev = [TsEvent.one(nm, float(t), ts.tinfo.units, ts.tinfo.start_date) for nm, t in zip(names, rel)]
        if len(ev) != len(names):
            raise _err("addevent needs one time for each event name.")
    else:
        raise _err("addevent needs a tsdata.event object, or event names and times.")
    new = ts.copy()
    new.events = new.events + ev
    return new


@_method("delevent")
def _m_delevent(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "delevent")
    if not rest:
        raise MatlabError("delevent needs the name of the event.", "MATLAB:minrhs")
    names = set(_texts(rest[0]) or [])
    occurrence = int(_scalar(rest[1], "occurrence")) if len(rest) > 1 else None
    new = ts.copy()
    if occurrence is None:
        new.events = [e for e in ts.events if e["Name"] not in names]
    else:
        seen = 0
        kept = []
        for e in ts.events:
            if e["Name"] in names:
                seen += 1
                if seen == occurrence:
                    continue
            kept.append(e)
        new.events = kept
    return new


def _event_time(interp, ts: TimeSeries, ev: Any, occurrence: Any) -> float:
    if isinstance(ev, TsEvent):
        e = ev.items[0] if ev.items else None
    else:
        name = text(ev, "event name")
        k = int(_scalar(occurrence, "occurrence")) if occurrence is not None else 1
        hits = [e for e in ts.events if e["Name"] == name]
        e = hits[k - 1] if 0 < k <= len(hits) else None
        if e is None:
            raise _err(f"The timeseries has no event called '{name}'" + (f" (occurrence {k})." if k > 1 else "."),
                       "MATLAB:timeseries:NoSuchEvent")
    if e is None:
        raise _err("The event is empty.")
    t = e["Time"] * UNITS[e["Units"]]
    if e["StartDate"] and ts.tinfo.start_date:
        t += (_start_num(interp, e["StartDate"]) - _start_num(interp, ts.tinfo.start_date)) * 86400.0
    return t / UNITS[ts.tinfo.units]


def _event_getter(name: str, test):
    def fn(interp, args, nargout):
        ts, rest = _ts_and_rest(args, name)
        if not rest:
            raise MatlabError(f"{name} needs an event.", "MATLAB:minrhs")
        t = _event_time(interp, ts, rest[0], rest[1] if len(rest) > 1 else None)
        return _subset(ts, np.nonzero(test(ts.time, t))[0])
    _METHODS[name] = fn


_event_getter("gettsafterevent", lambda tt, t: tt > t + TOL * max(1.0, abs(t)))
_event_getter("gettsafteratevent", lambda tt, t: tt >= t - TOL * max(1.0, abs(t)))
_event_getter("gettsatevent", lambda tt, t: _close(tt, t))
_event_getter("gettsbeforeevent", lambda tt, t: tt < t - TOL * max(1.0, abs(t)))
_event_getter("gettsbeforeatevent", lambda tt, t: tt <= t + TOL * max(1.0, abs(t)))


@_method("gettsbetweenevents")
def _m_gettsbetweenevents(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "gettsbetweenevents")
    if len(rest) < 2:
        raise MatlabError("gettsbetweenevents needs two events.", "MATLAB:minrhs")
    t1 = _event_time(interp, ts, rest[0], rest[2] if len(rest) > 2 else None)
    t2 = _event_time(interp, ts, rest[1], rest[3] if len(rest) > 3 else None)
    mask = (ts.time >= t1 - TOL * max(1.0, abs(t1))) & (ts.time <= t2 + TOL * max(1.0, abs(t2)))
    return _subset(ts, np.nonzero(mask)[0])


# -- statistics -------------------------------------------------------------------------------------------

def _time_weights(t: np.ndarray) -> np.ndarray:
    n = t.size
    if n < 2:
        return np.ones(n)
    w = np.empty(n)
    w[0] = t[1] - t[0]
    w[-1] = t[-1] - t[-2]
    if n > 2:
        w[1:-1] = (t[2:] - t[:-2]) / 2
    m = np.mean(w)
    return w / m if m > 0 else np.ones(n)


def _fill_missing(t: np.ndarray, x: np.ndarray, missing: np.ndarray) -> np.ndarray:
    good = ~missing
    if good.sum() == 0 or not missing.any():
        return x
    out = x.copy()
    if good.sum() == 1:
        out[missing] = x[good][0]
        return out
    out[missing] = np.interp(t[missing], t[good], x[good].real) + \
        (1j * np.interp(t[missing], t[good], x[good].imag) if np.iscomplexobj(x) else 0)
    return out


def _stat(name: str):
    def fn(interp, args, nargout):
        ts, rest = _ts_and_rest(args, name)
        opts = _name_value(rest, ("MissingData", "Quality", "Weighting"), name)
        missing = match_option(text(opts["MissingData"]), ("remove", "interpolate", "interpolation")) \
            if "MissingData" in opts else "remove"
        if missing is None:
            raise _err(f"{name}: 'MissingData' must be 'remove' or 'interpolate'.",
                       f"MATLAB:timeseries:{name}:BadOption")
        weighting = match_option(text(opts["Weighting"]), ("none", "time")) if "Weighting" in opts else "none"
        if weighting is None:
            raise _err(f"{name}: 'Weighting' must be 'none' or 'time'.", f"MATLAB:timeseries:{name}:BadOption")
        S = ts.samples()
        n = S.shape[0]
        out_shape = ts.sample_shape()
        flat = S.reshape(n, -1)
        flat = flat.astype(complex if np.iscomplexobj(flat) else float)
        mask = np.isnan(flat) if ts.treat_nan else np.zeros(flat.shape, dtype=bool)
        if "Quality" in opts and ts.quality is not None:
            codes = fnum(opts["Quality"]).reshape(-1)
            q = ts.quality_samples()
            qm = np.isin(q, codes)
            mask |= qm.reshape(n, -1) if qm.ndim > 1 else qm.reshape(-1, 1)
        w = _time_weights(ts.time) if weighting == "time" else None
        res = np.empty(flat.shape[1], dtype=flat.dtype)
        for j in range(flat.shape[1]):
            col, m = flat[:, j], mask[:, j]
            if missing != "remove":
                col = _fill_missing(ts.time, col, m)
                m = np.zeros(n, dtype=bool) if (~m).any() else m
            x = col[~m]
            if w is not None:
                x = x * w[~m]
            res[j] = _reduce(interp, name, x)
        if not np.iscomplexobj(res) or not np.any(res.imag):
            res = res.real if np.iscomplexobj(res) else res
        return res.reshape(out_shape)
    _METHODS[name] = fn


def _reduce(interp, name: str, x: np.ndarray) -> Any:
    if name == "sum":
        return np.sum(x) if x.size else 0.0
    if x.size == 0:
        return np.nan
    if name == "mean":
        return np.mean(x)
    if name == "median":
        return np.median(x)
    if name == "max":
        return np.max(x) if not np.iscomplexobj(x) else x[np.argmax(np.abs(x))]
    if name == "min":
        return np.min(x) if not np.iscomplexobj(x) else x[np.argmin(np.abs(x))]
    if name in ("std", "var"):
        if x.size == 1:
            return 0.0 if np.isfinite(x[0]) else np.nan
        v = np.var(x, ddof=1)
        return np.sqrt(v) if name == "std" else v
    return float(np.asarray(BUILTINS["iqr"](interp, [x.real.reshape(-1, 1)], 1)).reshape(-1)[0])


for _name in ("mean", "median", "max", "min", "std", "var", "sum", "iqr"):
    _stat(_name)


# -- signal-style methods ------------------------------------------------------------------------------

def _columns_arg(v: Any, ncols: int) -> np.ndarray:
    return _positions(v, ncols)


@_method("detrend")
def _m_detrend(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "detrend")
    kind = match_option(text(rest[0], "detrend method"), ("constant", "linear")) if rest else "linear"
    if kind is None:
        raise _err("detrend: the method must be 'constant' or 'linear'.", "MATLAB:timeseries:detrend:BadMethod")
    S = ts.samples()
    n = S.shape[0]
    flat = S.reshape(n, -1).astype(complex if np.iscomplexobj(S) else float)
    cols = _columns_arg(rest[1], flat.shape[1]) if len(rest) > 1 else np.arange(flat.shape[1])
    t = ts.time
    for j in cols:
        y = flat[:, j]
        good = ~np.isnan(y)
        if not good.any():
            continue
        if kind == "constant" or good.sum() < 2:
            flat[:, j] = y - np.mean(y[good])
        else:
            A = np.column_stack([t[good], np.ones(int(good.sum()))])
            coef, *_ = np.linalg.lstsq(A, y[good], rcond=None)
            flat[:, j] = y - (coef[0] * t + coef[1])
    return ts.with_samples(t, flat.reshape(S.shape), ts.quality_samples())


@_method("filter")
def _m_filter(interp, args, nargout):
    if not args or not isinstance(args[0], TimeSeries):
        raise _err("For a timeseries call filter(ts, b, a).", "MATLAB:timeseries:filter:BadInput")
    ts, rest = _ts_and_rest(args, "filter")
    if len(rest) < 2:
        raise MatlabError("filter needs the coefficients, filter(ts, b, a).", "MATLAB:minrhs")
    from scipy.signal import lfilter
    b = fnum(rest[0]).reshape(-1)
    a = fnum(rest[1]).reshape(-1)
    if a.size == 0 or a[0] == 0:
        raise _err("The first denominator coefficient must not be zero.", "MATLAB:filter:firstCoeffZero")
    S = ts.samples()
    n = S.shape[0]
    flat = S.reshape(n, -1).astype(complex if np.iscomplexobj(S) or np.iscomplexobj(b) else float)
    cols = _columns_arg(rest[2], flat.shape[1]) if len(rest) > 2 else np.arange(flat.shape[1])
    if n:
        flat[:, cols] = lfilter(b, a, flat[:, cols], axis=0)
    return ts.with_samples(ts.time, flat.reshape(S.shape), ts.quality_samples())


@_method("idealfilter")
def _m_idealfilter(interp, args, nargout):
    ts, rest = _ts_and_rest(args, "idealfilter")
    if len(rest) < 2:
        raise MatlabError("idealfilter needs the frequency interval and 'pass' or 'notch'.", "MATLAB:minrhs")
    band = fnum(rest[0]).reshape(-1, 2) if fnum(rest[0]).size % 2 == 0 else None
    if band is None:
        raise _err("The frequency interval must be [low high] pairs.", "MATLAB:timeseries:idealfilter:BadInterval")
    kind = match_option(text(rest[1], "filter type"), ("pass", "notch"))
    if kind is None:
        raise _err("The filter type must be 'pass' or 'notch'.", "MATLAB:timeseries:idealfilter:BadType")
    step = _increment(ts.time)
    if not np.isfinite(step):
        raise _err("idealfilter needs uniformly spaced times (use resample first).",
                   "MATLAB:timeseries:idealfilter:NonUniform")
    S = ts.samples()
    n = S.shape[0]
    flat = S.reshape(n, -1).astype(float)
    cols = _columns_arg(rest[2], flat.shape[1]) if len(rest) > 2 else np.arange(flat.shape[1])
    freqs = np.abs(np.fft.fftfreq(n, d=step))
    inband = np.zeros(n, dtype=bool)
    for lo, hi in band:
        inband |= (freqs >= lo) & (freqs <= hi)
    keep = inband if kind == "pass" else ~inband
    for j in cols:
        y = flat[:, j]
        mu = np.mean(y)
        spec = np.fft.fft(y - mu)
        spec[~keep] = 0
        flat[:, j] = np.real(np.fft.ifft(spec)) + (mu if kind == "notch" else 0.0)
    return ts.with_samples(ts.time, flat.reshape(S.shape), ts.quality_samples())


# -- arithmetic ---------------------------------------------------------------------------------------------

def _arith(op: str, left: Any, right: Any) -> TimeSeries:
    ts = left if isinstance(left, TimeSeries) else right
    if isinstance(left, TimeSeries) and isinstance(right, TimeSeries):
        if left.n != right.n or not np.allclose(left.time * UNITS[left.tinfo.units],
                                                right.time * UNITS[right.tinfo.units], rtol=0, atol=1e-9) \
                or left.tinfo.start_date != right.tinfo.start_date:
            raise _err("The timeseries must have the same times to be combined with an arithmetic operator "
                       "(use synchronize first).", "MATLAB:timeseries:arith:TimeMismatch")
        a, b = left.samples(), right.samples()
        if a.shape[1:] != b.shape[1:] and int(np.prod(a.shape[1:])) != 1 and int(np.prod(b.shape[1:])) != 1:
            raise _err("The timeseries samples must be the same size (or one of them scalar).",
                       "MATLAB:timeseries:arith:SizeMismatch")
        a2, b2 = _expand(a, b)
        S = _apply(op, a2, b2)
        return ts.with_samples(ts.time, S, None)
    other = right if ts is left else left
    if isinstance(other, MObject) or not isinstance(other, np.ndarray):
        raise _err(f"Operator '{op}' needs a timeseries and a number (or another timeseries), not "
                   f"'{class_name(other)}'.", "MATLAB:UndefinedFunction")
    x = np.asarray(other)
    S = ts.samples()
    if op == "*" and ts is left and x.ndim == 2 and x.size > 1 and ts.first:
        return ts.with_samples(ts.time, ops.matmul(ts.data, x), None)
    if op in ("*", "/", "\\") and x.size != 1:
        raise _err(f"Operator '{op}' with a timeseries needs a scalar; use '.{op}' for elementwise work.",
                   "MATLAB:timeseries:arith:SizeMismatch")
    if x.size == 1:
        xs = x.reshape((1,) * S.ndim)
    elif x.shape == as_2d(ts.data).shape:
        xs = ts._samples_of(x) if hasattr(ts, "_samples_of") else _layout_samples(ts, x)
    elif tuple(x.shape) == tuple(ts.sample_shape()) or x.size == int(np.prod(S.shape[1:])):
        xs = x.reshape((1,) + S.shape[1:], order="F" if not ts.first else "C")
    else:
        raise _err(f"The array ({size_str(x.shape)}) must be a scalar, one sample ({size_str(ts.sample_shape())}) "
                   f"or the size of the data ({size_str(as_2d(ts.data).shape)}).",
                   "MATLAB:timeseries:arith:SizeMismatch")
    a, b = (S, xs) if ts is left else (xs, S)
    a2, b2 = _expand(a, b)
    return ts.with_samples(ts.time, _apply(op, a2, b2), None)


def _layout_samples(ts: TimeSeries, x: np.ndarray) -> np.ndarray:
    full = x.reshape(ts.data.shape)
    return full if ts.first else np.moveaxis(full, -1, 0)


def _expand(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.ndim < b.ndim:
        a = a.reshape(a.shape + (1,) * (b.ndim - a.ndim))
    if b.ndim < a.ndim:
        b = b.reshape(b.shape + (1,) * (a.ndim - b.ndim))
    return a, b


def _apply(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    elem = {"*": ".*", "/": "./", "\\": ".\\"}.get(op, op)
    shape = np.broadcast_shapes(a.shape, b.shape)
    a3 = np.broadcast_to(a, shape).reshape(shape[0], -1)
    b3 = np.broadcast_to(b, shape).reshape(shape[0], -1)
    return np.asarray(ops.elementwise(elem, np.ascontiguousarray(a3), np.ascontiguousarray(b3))).reshape(shape)


# -- plotting ------------------------------------------------------------------------------------------

@_method("plot")
def _m_plot(interp, args, nargout):
    from .plotting import GraphicsHandle
    rest = list(args)
    ax = [rest.pop(0)] if rest and isinstance(rest[0], GraphicsHandle) else []
    if not rest or not isinstance(rest[0], TimeSeries):
        raise _err("plot of a timeseries: plot(ts) or plot(ts, LineSpec, ...).", "MATLAB:timeseries:plot:BadInput")
    ts = rest.pop(0)
    if any(isinstance(r, TimeSeries) for r in rest):
        raise _err("Plot one timeseries per call (use hold on to add more).", "MATLAB:timeseries:plot:BadInput")
    if ts.tinfo.start_date:
        from .datetimes import DatetimeArray
        base = _start_num(interp, ts.tinfo.start_date)
        secs = base * 86400.0 + ts.time * UNITS[ts.tinfo.units]
        x: Any = DatetimeArray(secs.reshape(-1, 1))
        xlabel = "Time"
    else:
        x = ts.time_value()
        xlabel = f"Time ({ts.tinfo.units})"
    S = ts.samples()
    y = S.reshape(S.shape[0], -1)
    y = y.astype(float) if y.dtype.kind in "biu" else y
    fn = "stairs" if ts.dinfo.interp.name == "zoh" and not ts.tinfo.start_date else "plot"
    h = interp.call_builtin(fn, BUILTINS[fn], ax + [x, as_2d(np.ascontiguousarray(y))] + rest, 1)
    name = ts.name
    ylabel = name + (f" ({ts.dinfo.units})" if ts.dinfo.units else "")
    for f, label in (("title", f"Time Series Plot:{name}"), ("xlabel", xlabel), ("ylabel", ylabel)):
        interp.call_builtin(f, BUILTINS[f], ax + [char(label)], 0)
    return h[0] if nargout and h else None


# -- conversions ----------------------------------------------------------------------------------------

_DURATION_FORMAT = {"weeks": "d", "days": "d", "hours": "h", "minutes": "m", "seconds": "s",
                    "milliseconds": "s", "microseconds": "s", "nanoseconds": "s"}


@builtin("timeseries2timetable")
def _timeseries2timetable(interp, args, nargout):
    from .datetimes import DatetimeArray, DurationArray
    from .timetables import TimetableObject
    if not args:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    series = []
    for a in args:
        if isinstance(a, Cell):
            series += list(a.values_f())
        else:
            series.append(a)
    series = [_need_ts(s, "timeseries2timetable") for s in series]
    first = series[0]
    secs = first.time * UNITS[first.tinfo.units]
    for s in series[1:]:
        other = s.time * UNITS[s.tinfo.units]
        if other.size != secs.size or not np.allclose(other, secs, rtol=0, atol=1e-9) or \
                s.tinfo.start_date != first.tinfo.start_date:
            raise _err("All of the timeseries must have the same sample times.",
                       "MATLAB:timeseries2timetable:DifferentTimes")
    if first.tinfo.start_date:
        base = _start_num(interp, first.tinfo.start_date)
        fmt = None
        times: Any = DatetimeArray((base * 86400.0 + secs).reshape(-1, 1), fmt)
    else:
        times = DurationArray(secs.reshape(-1, 1), _DURATION_FORMAT[first.tinfo.units])
    unnamed = sum(1 for s in series if s.name in ("unnamed", ""))
    names, cols, units = [], [], {}
    k = 0
    for s in series:
        if s.name in ("unnamed", ""):
            k += 1
            nm = "Data" if unnamed == 1 else f"Data_{k}"
        else:
            nm = s.name
        base_nm, j = nm, 1
        while nm in names:
            j += 1
            nm = f"{base_nm}_{j}"
        S = s.samples()
        col = S if S.ndim > 1 else S.reshape(-1, 1)
        names.append(nm)
        cols.append(as_2d(np.ascontiguousarray(col)))
        if s.dinfo.units:
            units[nm] = s.dinfo.units
        if s.quality is not None:
            interp.warn(f"The quality codes of timeseries '{s.name}' are not carried into the timetable.",
                        "MATLAB:timeseries2timetable:QualityIgnored")
    tt = TimetableObject(names, cols, times)
    tt.units = units
    if first.user_data is not None:
        tt.user_data = first.user_data
    return tt


_FORMAT_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


@builtin("timetable2timeseries")
def _timetable2timeseries(interp, args, nargout):
    from .timetables import TimetableObject
    if not args or not isinstance(args[0], TimetableObject):
        raise _err("timetable2timeseries needs a timetable.", "MATLAB:timetable2timeseries:InvalidInput")
    tt = args[0]
    names = list(tt.names)
    if len(args) > 1:
        wanted = _texts(args[1])
        if wanted is None:
            raise _err("The variable names must be text.")
        for w in wanted:
            if w not in names:
                raise MatlabError(f"Unrecognized table variable name '{w}'.", "MATLAB:table:UnrecognizedVarName")
        names = wanted
    rt = tt.row_times
    if rt.kind == "datetime":
        secs = np.asarray(rt.data, dtype=float).reshape(-1)
        base_secs = float(np.min(secs)) if secs.size else 0.0
        start = _datestr(interp, base_secs / 86400.0) if secs.size else ""
        units = "seconds"
        rel = secs - base_secs
    else:
        units = _FORMAT_UNITS.get(rt.fmt or "", "seconds")
        rel = np.asarray(rt.data, dtype=float).reshape(-1) / UNITS[units]
        start = ""
    out = []
    for nm in names:
        col = tt.columns[tt.names.index(nm)]
        if not isinstance(col, np.ndarray) or col.dtype.kind not in "biufc":
            raise _err(f"Variable '{nm}' is not numeric or logical, so it cannot become timeseries data.",
                       "MATLAB:timetable2timeseries:NonNumeric")
        ts = TimeSeries(col.copy(), rel.copy(), True, nm)
        ts.tinfo.units = units
        ts.tinfo.start_date = start
        ts.dinfo.units = tt.units.get(nm, "")
        out.append(ts)
    if nargout <= 1 and len(out) > 1:
        if not all(c.data.ndim == 2 for c in out):
            raise _err("Several variables need several outputs: [ts1, ts2] = timetable2timeseries(TT).")
        combined = TimeSeries(np.concatenate([c.data.astype(np.result_type(*[d.data for d in out]))
                                              for c in out], axis=1), rel.copy(), True, "unnamed")
        combined.tinfo.units, combined.tinfo.start_date = units, start
        return combined
    if not out:
        raise _err("The timetable has no variables.")
    return tuple(out[:max(nargout, 1)]) if nargout > 1 else out[0]


# -- MAT-files ---------------------------------------------------------------------------------------------

def to_saved(ts: TimeSeries) -> dict:
    """Fields of the struct a MAT-file holds for a timeseries."""
    def opt(v):
        return v if v is not None else _empty()
    ev = Cell.from_list([Struct.scalar({"name": char(e["Name"]), "time": np.array([[e["Time"]]]),
                                        "units": char(e["Units"]), "startdate": char(e["StartDate"]),
                                        "eventdata": opt(e["EventData"])}) for e in ts.events]) \
        if ts.events else Cell(np.empty((0, 0), dtype=object))
    return {
        "name": char(ts.name), "data": ts.data, "time": ts.time.reshape(-1, 1),
        "timefirst": bool_scalar(ts.first), "quality": opt(ts.quality),
        "units": char(ts.tinfo.units), "startdate": char(ts.tinfo.start_date), "format": char(ts.tinfo.fmt),
        "timeuserdata": opt(ts.tinfo.user_data), "dataunits": char(ts.dinfo.units),
        "interp": char(ts.dinfo.interp.name),
        "fhandle": ts.dinfo.interp.fhandle if ts.dinfo.interp.fhandle is not None else _empty(),
        "datauserdata": opt(ts.dinfo.user_data),
        "qualitycode": ts.qinfo.code.reshape(1, -1) if ts.qinfo.code.size else _empty(),
        "qualitydesc": Cell.from_list([char(d) for d in ts.qinfo.desc]),
        "qualityuserdata": opt(ts.qinfo.user_data), "events": ev, "userdata": opt(ts.user_data),
        "treatnan": bool_scalar(ts.treat_nan),
    }


def from_saved(d: dict) -> TimeSeries:
    def opt(v):
        return None if isinstance(v, np.ndarray) and v.size == 0 and v.dtype.kind == "f" else v

    def txt(v):
        return to_str(v) if isinstance(v, np.ndarray) and v.size else ""
    time = np.asarray(d["time"], dtype=float).reshape(-1)
    first = bool(np.asarray(d["timefirst"]).any())
    data = np.asarray(d["data"])
    n = time.size
    if not first and (data.ndim < 3 or data.shape[-1] != n):
        data = data.reshape(data.shape + (1,) * max(0, 3 - data.ndim)) if n == 1 else data
    ts = TimeSeries(data, time, first, txt(d["name"]))
    q = d.get("quality")
    ts.quality = q.astype(np.int8) if isinstance(q, np.ndarray) and q.size else None
    if ts.quality is not None and not first and ts.quality.shape != data.shape and ts.quality.size == data.size:
        ts.quality = ts.quality.reshape(data.shape)
    ts.tinfo.units = txt(d["units"]) or "seconds"
    ts.tinfo.start_date = txt(d["startdate"])
    ts.tinfo.fmt = txt(d["format"])
    ts.tinfo.user_data = opt(d.get("timeuserdata"))
    ts.dinfo.units = txt(d["dataunits"])
    fh = d.get("fhandle")
    ts.dinfo.interp = TsInterp(txt(d["interp"]) or "linear", fh if isinstance(fh, FunctionHandle) else None)
    ts.dinfo.user_data = opt(d.get("datauserdata"))
    code = d.get("qualitycode")
    ts.qinfo = TsQualityInfo(np.asarray(code, dtype=float).reshape(-1) if isinstance(code, np.ndarray) and code.size
                             else None, _texts(d["qualitydesc"]) if isinstance(d.get("qualitydesc"), Cell) else [],
                             opt(d.get("qualityuserdata")))
    events = []
    ev = d.get("events")
    if isinstance(ev, Cell):
        for e in ev.values_f():
            if isinstance(e, Struct) and e.data.size:
                f = e.data.flat[0]
                events.append(TsEvent.one(txt(f["name"]), float(np.asarray(f["time"]).reshape(-1)[0]),
                                          txt(f["units"]) or "seconds", txt(f["startdate"]), opt(f["eventdata"])))
    ts.events = events
    ts.user_data = opt(d.get("userdata"))
    ts.treat_nan = bool(np.asarray(d.get("treatnan", np.array([[True]]))).any())
    return ts


# -- functions that take a timeseries first ----------------------------------------------------------------

def _dispatcher(name: str):
    def fn(interp, args, nargout):
        if not args or not isinstance(args[0], TimeSeries):
            got = class_name(args[0]) if args else "none"
            raise MatlabError(f"Undefined function '{name}' for input arguments of type '{got}'. It is a method "
                              "of timeseries objects.", "MATLAB:UndefinedFunction")
        return _METHODS[name](interp, args, nargout)
    builtin(name)(fn)


for _name in ("getdatasamples", "getsamples", "getsampleusingtime", "getdatasamplesize", "getinterpmethod",
              "setinterpmethod", "addsample", "delsample", "addevent", "delevent", "getabstime", "setabstime",
              "setuniformtime", "getqualitydesc", "idealfilter", "gettsafterevent", "gettsafteratevent",
              "gettsatevent", "gettsbeforeevent", "gettsbeforeatevent", "gettsbetweenevents"):
    _dispatcher(_name)
