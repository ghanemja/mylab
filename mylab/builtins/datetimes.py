"""Dates and times: datetime, duration and calendarDuration arrays, plus the serial-date functions.

A :class:`DatetimeArray` holds an N-D array of points in time as seconds since
1970-01-01 00:00:00 on its own wall clock (NaN is NaT). A :class:`DurationArray`
holds lengths of time in seconds. One object is a whole array, so indexing,
concatenation and arithmetic work element-wise without object arrays.

Serial date numbers (``datenum``) count days from year 0: ``seconds / 86400 + 719529``.
"""

from __future__ import annotations

import math
import re
import time
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from .. import indexing as ix
from .. import ops
from ..display import TIMES
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, MString, StringArray, as_2d, char, is_char, string_result, to_matlab, to_numeric, to_str,
)
from ._registry import builtin
from ._util import bool_scalar, is_text, text

DAY = 86400.0
EPOCH_DATENUM = 719529.0  # datenum of 1970-01-01
YEAR_SECONDS = 365.2425 * DAY
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
DEFAULT_DATE = "dd-MMM-yyyy"
DEFAULT_FULL = "dd-MMM-yyyy HH:mm:ss"

_NATIVE: dict[str, Callable] = {}


def native(*names: str):
    def deco(fn: Callable) -> Callable:
        for n in names:
            _NATIVE[n] = fn
        return fn
    return deco


# -- calendar arithmetic (proleptic Gregorian, vectorized) ------------------------------------------

def days_from_civil(y, m, d):
    """Days since 1970-01-01 for year/month/day arrays (months must be 1..12)."""
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)
    y = y - (m <= 2)
    era = np.floor(y / 400)
    yoe = y - era * 400
    doy = np.floor((153 * (m + np.where(m > 2, -3, 9)) + 2) / 5) + np.asarray(d, dtype=float) - 1
    doe = yoe * 365 + np.floor(yoe / 4) - np.floor(yoe / 100) + doy
    return era * 146097 + doe - 719468


def civil_from_days(z):
    z = np.asarray(z, dtype=float) + 719468
    era = np.floor(z / 146097)
    doe = z - era * 146097
    yoe = np.floor((doe - np.floor(doe / 1460) + np.floor(doe / 36524) - np.floor(doe / 146096)) / 365)
    y = yoe + era * 400
    doy = doe - (365 * yoe + np.floor(yoe / 4) - np.floor(yoe / 100))
    mp = np.floor((5 * doy + 2) / 153)
    d = doy - np.floor((153 * mp + 2) / 5) + 1
    m = mp + np.where(mp < 10, 3, -9)
    return y + (m <= 2), m, d


def fields_from_seconds(s):
    """(year, month, day, hour, minute, second) arrays; NaN where the input is NaN."""
    s = np.asarray(s, dtype=float)
    with np.errstate(invalid="ignore"):
        days = np.floor(s / DAY)
        sod = s - days * DAY
        y, m, d = civil_from_days(days)
        h = np.floor(sod / 3600)
        mi = np.floor((sod - h * 3600) / 60)
        sec = sod - h * 3600 - mi * 60
    return y, m, d, h, mi, sec


def seconds_from_fields(y, m, d, h=0.0, mi=0.0, s=0.0):
    """Seconds since 1970 from components; out-of-range months, days and times roll over."""
    y, m, d, h, mi, s = (np.asarray(x, dtype=float) for x in (y, m, d, h, mi, s))
    with np.errstate(invalid="ignore"):
        y = y + np.floor((m - 1) / 12)
        m = np.mod(m - 1, 12) + 1
        return (days_from_civil(y, m, 1) + (d - 1)) * DAY + h * 3600 + mi * 60 + s


def is_leap(y):
    y = np.asarray(y, dtype=float)
    return ((np.mod(y, 4) == 0) & (np.mod(y, 100) != 0)) | (np.mod(y, 400) == 0)


def month_length(y, m):
    y, m = np.broadcast_arrays(np.asarray(y, dtype=float), np.asarray(m, dtype=float))
    base = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    mm = np.where(np.isfinite(m), np.mod(m - 1, 12), 0).astype(int)
    out = base[mm] + ((mm == 1) & is_leap(y))
    return np.where(np.isfinite(m) & np.isfinite(y), out, np.nan)


def weekday_index(s):
    """0 = Sunday ... 6 = Saturday (NaN for NaT)."""
    with np.errstate(invalid="ignore"):
        return np.mod(np.floor(np.asarray(s, dtype=float) / DAY) + 4, 7)


def add_months(s, n):
    """Shift by whole calendar months, clamping the day to the end of shorter months."""
    y, m, d, h, mi, sec = fields_from_seconds(s)
    n = np.asarray(n, dtype=float)
    total = (m - 1) + n
    ny = y + np.floor(total / 12)
    nm = np.mod(total, 12) + 1
    nd = np.minimum(d, month_length(ny, nm))
    return seconds_from_fields(ny, nm, nd, h, mi, sec)


def local_offset(posix: float | None = None) -> float:
    """Seconds the local wall clock is ahead of UTC at the given instant."""
    try:
        return float(time.localtime(time.time() if posix is None else posix).tm_gmtoff)
    except (OverflowError, OSError, ValueError):
        return float(time.localtime().tm_gmtoff)


def now_seconds(zone: str = "") -> float:
    t = time.time()
    return t if zone == "UTC" else t + local_offset(t)


def _zone(name: str) -> str:
    key = name.strip()
    if key == "":
        return ""
    if key.upper() in ("UTC", "Z", "GMT", "ETC/UTC", "UTC+0", "+00:00"):
        return "UTC"
    if key.lower() == "local":
        return "local"
    raise MatlabError(f"Time zone '{name}' is not supported: only 'UTC' and 'local' are available "
                      "(or '' for no time zone).", "MATLAB:datetime:UnsupportedTimeZone")


# -- format strings ------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _tokens(fmt: str) -> tuple:
    """Split a format into ('lit', text) and (letter, count) tokens."""
    out: list[tuple] = []
    i, n = 0, len(fmt)
    while i < n:
        c = fmt[i]
        if c == "'":
            j = i + 1
            buf = ""
            if j < n and fmt[j] == "'":
                out.append(("lit", "'"))
                i = j + 1
                continue
            while j < n:
                if fmt[j] == "'":
                    if j + 1 < n and fmt[j + 1] == "'":
                        buf += "'"
                        j += 2
                        continue
                    break
                buf += fmt[j]
                j += 1
            out.append(("lit", buf))
            i = j + 1
        elif c.isalpha() and c.isascii():
            j = i
            while j < n and fmt[j] == c:
                j += 1
            out.append((c, j - i))
            i = j
        else:
            out.append(("lit", c))
            i += 1
    return tuple(out)


def _has_time(fmt: str) -> bool:
    return any(t[0] in "HhmsSa" for t in _tokens(fmt) if t[0] != "lit")


def _offset_text(zone: str, s: float, colon: bool) -> str:
    off = 0.0 if zone != "local" else local_offset(s - local_offset())
    sign = "-" if off < 0 else "+"
    off = abs(off)
    hh, mm = int(off // 3600), int(off % 3600 // 60)
    return f"{sign}{hh:02d}:{mm:02d}" if colon else f"{sign}{hh:02d}{mm:02d}"


def format_datetime(s: float, fmt: str, zone: str = "") -> str:
    if s != s:
        return "NaT"
    if math.isinf(s):
        return "Inf" if s > 0 else "-Inf"
    frac_digits = max([t[1] for t in _tokens(fmt) if t[0] == "S"], default=0)
    # Guard against 59.9999999 from floating point: snap to the displayed resolution.
    q = 10.0 ** max(frac_digits, 6)
    s = math.floor(s * q + 0.5) / q
    y, mo, d, h, mi, sec = (float(x) for x in fields_from_seconds(s))
    out = []
    for kind, n in _tokens(fmt):
        if kind == "lit":
            out.append(n)
        elif kind in "yu":
            yi = int(y)
            out.append(f"{yi % 100:02d}" if n == 2 else (f"{yi:0{n}d}" if yi >= 0 else f"-{-yi:0{n}d}"))
        elif kind == "M" or kind == "L":
            mi_ = int(mo)
            out.append(str(mi_) if n == 1 else f"{mi_:02d}" if n == 2 else MONTHS[mi_ - 1][:3] if n == 3
                       else MONTHS[mi_ - 1] if n == 4 else MONTHS[mi_ - 1][0])
        elif kind == "d":
            out.append(f"{int(d):0{n}d}")
        elif kind == "D":
            doy = int(days_from_civil(y, mo, d) - days_from_civil(y, 1, 1)) + 1
            out.append(f"{doy:0{n}d}")
        elif kind in "eE":
            w = int(weekday_index(s))
            if kind == "e" and n <= 2:
                out.append(f"{w + 1:0{n}d}")
            else:
                out.append(WEEKDAYS[w][:3] if n <= 3 else WEEKDAYS[w] if n == 4 else WEEKDAYS[w][0])
        elif kind == "H":
            out.append(f"{int(h):0{n}d}")
        elif kind == "h":
            out.append(f"{(int(h) + 11) % 12 + 1:0{n}d}")
        elif kind == "m":
            out.append(f"{int(mi):0{n}d}")
        elif kind == "s":
            out.append(f"{int(math.floor(sec + 1e-9)):0{n}d}")
        elif kind == "S":
            frac = sec - math.floor(sec + 1e-9)
            out.append(f"{int(math.floor(max(frac, 0.0) * 10 ** n + 1e-6)):0{n}d}"[:n])
        elif kind == "a":
            out.append("AM" if h < 12 else "PM")
        elif kind == "Q":
            qn = (int(mo) - 1) // 3 + 1
            out.append(str(qn) if n < 3 else f"Q{qn}")
        elif kind == "w":
            out.append(f"{int(_week_of_year(np.array(s))):0{n}d}")
        elif kind == "Z":
            out.append(_offset_text(zone, s, n >= 5) if zone else "")
        elif kind in "xX":
            out.append("Z" if (kind == "X" and zone == "UTC") else (_offset_text(zone, s, n >= 3) if zone else ""))
        elif kind == "z":
            out.append("UTC" if zone == "UTC" else (time.tzname[0] if zone == "local" else ""))
        else:
            out.append(kind * n)
    return "".join(out)


@lru_cache(maxsize=256)
def _parse_regex(fmt: str):
    parts, keys = [], []
    toks = _tokens(fmt)
    numeric = [t[0] != "lit" and (t[0] in "yudDHhmsS" or (t[0] in "ML" and t[1] <= 2)) for t in toks]
    for pos, (kind, n) in enumerate(toks):
        glued = (pos > 0 and numeric[pos - 1]) or (pos + 1 < len(toks) and numeric[pos + 1])
        two = r"(\d{2})" if glued else r"(\d{1,2})"
        if kind == "lit":
            parts.append(r"\s+" if n.isspace() else re.escape(n))
            continue
        if kind in "yu":
            pat = r"(\d{4})" if n == 4 else r"(\d{2})" if n == 2 else r"(-?\d+)"
            keys.append("yy" if n == 2 else "y")
        elif kind in "ML":
            pat = two if n == 2 else r"(\d{1,2})" if n == 1 else r"([A-Za-z]{3})" if n == 3 else r"([A-Za-z]+)"
            keys.append("M" if n <= 2 else "Mname")
        elif kind == "d":
            pat = two if n == 2 else r"(\d{1,2})"
            keys.append("d")
        elif kind == "D":
            pat = r"(\d{1,3})"
            keys.append("D")
        elif kind in "eE":
            pat = r"([A-Za-z]+)" if n >= 3 or kind == "E" else r"(\d{1,2})"
            keys.append("skip")
        elif kind in "Hh":
            pat = two if n == 2 else r"(\d{1,2})"
            keys.append(kind)
        elif kind == "m":
            pat = two if n == 2 else r"(\d{1,2})"
            keys.append("m")
        elif kind == "s":
            pat = two if n == 2 else r"(\d{1,2})"
            keys.append("s")
        elif kind == "S":
            pat = r"(\d{1,9})"
            keys.append("S")
        elif kind == "a":
            pat = r"([AaPp]\.?[Mm]\.?)"
            keys.append("a")
        elif kind in "ZxX":
            pat = r"(Z|[+-]\d{2}:?\d{2})"
            keys.append("Z")
        elif kind == "z":
            pat = r"([A-Za-z]+)"
            keys.append("skip")
        elif kind == "Q":
            pat = r"(Q?\d)"
            keys.append("skip")
        else:
            raise MatlabError(f"Unsupported letter '{kind}' in the date format '{fmt}'.")
        parts.append(pat)
    return re.compile(r"\s*" + "".join(parts) + r"\s*$"), tuple(keys)


def _month_number(name: str) -> int | None:
    key = name.lower()
    for i, full in enumerate(MONTHS):
        if key == full.lower() or key == full[:3].lower():
            return i + 1
    if key == "sept":
        return 9
    return None


def parse_with_format(s: str, fmt: str) -> float:
    """Seconds since 1970 for text in the given format; NaN when it does not match."""
    rx, keys = _parse_regex(fmt)
    m = rx.match(s)
    if not m:
        return float("nan")
    today = fields_from_seconds(now_seconds())
    y, mo, d, h, mi, sec = None, None, None, 0.0, 0.0, 0.0
    pm = None
    doy = None
    twelve = False
    for key, val in zip(keys, m.groups()):
        if key == "y":
            y = float(val)
        elif key == "yy":
            v = int(val)
            y = float(2000 + v if v < 50 else 1900 + v)
        elif key == "M":
            mo = float(val)
        elif key == "Mname":
            num = _month_number(val)
            if num is None:
                return float("nan")
            mo = float(num)
        elif key == "d":
            d = float(val)
        elif key == "D":
            doy = float(val)
        elif key == "H":
            h = float(val)
        elif key == "h":
            h, twelve = float(val), True
        elif key == "m":
            mi = float(val)
        elif key == "s":
            sec += float(val)
        elif key == "S":
            sec += float("0." + val)
        elif key == "a":
            pm = val.lower().startswith("p")
    if twelve or pm is not None:
        if h < 1 or h > 12:
            return float("nan")
        h = h % 12 + (12 if pm else 0)
    if mo is not None and not 1 <= mo <= 12:
        return float("nan")
    if h > 24 or mi > 59 or sec >= 61:
        return float("nan")
    if y is None and mo is None and d is None and doy is None:
        y, mo, d = float(today[0]), float(today[1]), float(today[2])
    if y is None:
        y = float(today[0])
    if doy is not None:
        return float(seconds_from_fields(y, 1, doy, h, mi, sec))
    mo = 1.0 if mo is None else mo
    d = 1.0 if d is None else d
    if not 1 <= d <= float(month_length(y, mo)):
        return float("nan")
    return float(seconds_from_fields(y, mo, d, h, mi, sec))


_ISO = re.compile(r"\s*(-?\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(\.\d+)?)?)?\s*"
                  r"(Z|[+-]\d{2}:?\d{2})?\s*$")

_AUTO_FORMATS = [
    "dd-MMM-yyyy HH:mm:ss", "dd-MMM-yyyy HH:mm", "dd-MMM-yyyy", "yyyy/MM/dd HH:mm:ss", "yyyy/MM/dd HH:mm",
    "yyyy/MM/dd", "MM/dd/yyyy HH:mm:ss", "MM/dd/yyyy hh:mm:ss a", "MM/dd/yyyy HH:mm", "MM/dd/yyyy hh:mm a",
    "MM/dd/yyyy", "dd/MM/yyyy HH:mm:ss", "dd/MM/yyyy HH:mm", "dd/MM/yyyy", "dd.MM.yyyy HH:mm:ss", "dd.MM.yyyy HH:mm",
    "dd.MM.yyyy", "MMM dd, yyyy HH:mm:ss", "MMM dd, yyyy", "MMMM dd, yyyy HH:mm:ss", "MMMM dd, yyyy",
    "MMM dd, yyyy HH:mm", "MMMM dd, yyyy HH:mm", "dd MMM yyyy HH:mm:ss", "dd MMM yyyy HH:mm", "dd MMM yyyy",
    "dd MMMM yyyy HH:mm:ss", "dd MMMM yyyy HH:mm", "dd MMMM yyyy", "MMM-dd-yyyy", "MMM dd yyyy",
    "yyyyMMdd'T'HHmmss", "yyyy-MM-dd HH:mm:ss.S", "MM/dd/yy", "HH:mm:ss.S", "HH:mm:ss", "hh:mm:ss a",
    "hh:mm a", "HH:mm", "MMM yyyy", "MMMM yyyy", "MMM-yyyy", "yyyy-MM", "yyyyMMdd",
    "eee MMM dd HH:mm:ss yyyy", "eee, dd MMM yyyy HH:mm:ss",
]


def _parse_iso(s: str) -> tuple[float, bool] | None:
    m = _ISO.match(s)
    if not m:
        return None
    y, mo, d = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if not 1 <= mo <= 12 or not 1 <= d <= float(month_length(y, mo)):
        return (float("nan"), False)
    h = float(m.group(4) or 0)
    mi = float(m.group(5) or 0)
    sec = float(m.group(6) or 0) + float(m.group(7) or 0)
    return float(seconds_from_fields(y, mo, d, h, mi, sec)), m.group(4) is not None


def parse_texts(items: list, fmt: str | None) -> tuple[np.ndarray, bool]:
    """Parse a list of strings (None/'' -> NaT). Returns seconds and whether a time of day was present."""
    out = np.full(len(items), np.nan)
    live = [(k, s.strip()) for k, s in enumerate(items)
            if s is not None and s.strip() != "" and s.strip().lower() not in ("nat", "<missing>")]
    if fmt is not None:
        for k, s in live:
            out[k] = parse_with_format(s, fmt)
            if out[k] != out[k]:
                raise MatlabError(f"Unable to convert '{s}' to datetime using the format '{fmt}'.",
                                  "MATLAB:datetime:ParseErr")
        return out, _has_time(fmt)
    if not live:
        return out, False
    iso = [_parse_iso(s) for _, s in live]
    if all(r is not None for r in iso):
        for (k, _), r in zip(live, iso):
            out[k] = r[0]
        return out, any(r[1] for r in iso)
    best, best_count = None, 0
    for cand in _AUTO_FORMATS:
        vals = [parse_with_format(s, cand) for _, s in live]
        good = sum(1 for v in vals if v == v)
        if good > best_count:
            best, best_count = (cand, vals), good
            if good == len(live):
                break
    if best is None:
        raise MatlabError(f"Could not recognize the date/time format of '{live[0][1]}'. Use the 'InputFormat' "
                          "option, for example datetime(text, 'InputFormat', 'dd.MM.yyyy HH:mm').",
                          "MATLAB:datetime:UnrecognizedDateStringSuggestLocale")
    for (k, _), v in zip(live, best[1]):
        out[k] = v
    return out, _has_time(best[0])


def _text_items(v: Any) -> tuple[list, tuple] | None:
    """Strings and their array shape for char / string / cellstr input; None if v is not text."""
    if isinstance(v, MString):
        return [str(v)], (1, 1)
    if isinstance(v, StringArray):
        return list(v.data.flatten(order="F")), v.data.shape
    if is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return ["".join(r) for r in v], (v.shape[0], 1)
        return [to_str(v)], (1, 1)
    if isinstance(v, Cell) and all(is_char(x) or isinstance(x, MString) for x in v.data.flat):
        return [to_str(x) for x in v.data.flatten(order="F")], v.data.shape
    return None


# -- the array classes -------------------------------------------------------------------------------------

def _size_text(shape: tuple) -> str:
    return TIMES.join(str(d) for d in shape)


class _TimeArray(MObject):
    """Shared behaviour of datetime and duration arrays: shape, indexing, concatenation, display."""

    kind = "time"
    fill_value = float("nan")  # what new elements get when an assignment grows the array

    def __init__(self, data: Any, fmt: str | None = None):
        a = np.asarray(data, dtype=float)
        self.data = as_2d(a) if a.ndim != 2 else a
        self.fmt = fmt

    # shape and duck-typed array attributes (used by for-loops)
    def matlab_shape(self) -> tuple:
        return tuple(self.data.shape)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def like(self, data: Any) -> "_TimeArray":
        raise NotImplementedError

    def copy(self) -> "_TimeArray":
        return self.like(self.data.copy())

    def reshape(self, shape, order="F") -> "_TimeArray":
        return self.like(self.data.reshape(shape, order=order))

    def __getitem__(self, key) -> "_TimeArray":
        return self.like(self.data[key])

    def class_name(self) -> str:
        return self.kind

    def isa_class(self, name: str) -> bool:
        return name == self.kind

    def has_method(self, name: str) -> bool:
        return name in _NATIVE

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        res = _NATIVE[name](interp, args, nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    def numeric_error(self, what: str) -> str:
        return (f"A {self.kind} array cannot be used as a number here. Convert it first (for example with "
                f"{'datenum or posixtime' if self.kind == 'datetime' else 'seconds, minutes, hours or days'}); "
                f"for a range of {self.kind} values write colon(first, step, last).")

    def object_equals(self, other: Any) -> bool:
        if type(other) is not type(self):
            return False
        a, b = self.data, other.data
        return a.shape == b.shape and bool(np.all(a == b))

    def coerce(self, value: Any) -> np.ndarray:
        """Seconds array for a value assigned into / concatenated with this array."""
        raise NotImplementedError

    # indexing
    def index_get(self, interp, args: list) -> Any:
        return self.like(ix.get(self.data, list(args)))

    def index_set(self, interp, args: list, value: Any) -> Any:
        vals = self.coerce(value)
        return self.like(ix.set_(self.data.copy(), list(args), vals, lambda: self.fill_value))

    def index_delete(self, interp, args: list) -> Any:
        return self.like(ix.delete(self.data, list(args)))

    def brace_get(self, interp, args: list) -> list:
        raise MatlabError(f"Brace indexing is not supported for variables of type '{self.kind}'. "
                          "Use parentheses: t(k).")

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        first = next(i for i in items if isinstance(i, _TimeArray))
        for i in items:
            if isinstance(i, _TimeArray) and type(i) is not type(first):
                raise MatlabError(f"Cannot concatenate a {first.kind} array with a {i.kind} array.")
        datas = []
        for i in items:
            if isinstance(i, np.ndarray) and i.size == 0 and not is_char(i):
                continue
            datas.append(i.data if isinstance(i, _TimeArray) else first.coerce(i))
        return first.like(ops._concat_arrays(datas, axis))

    # display
    def element_texts(self) -> np.ndarray:
        raise NotImplementedError

    def summary_text(self, in_cell: bool) -> str:
        if self.data.size == 1:
            s = str(self.element_texts().flat[0])
            return f"[{s}]" if in_cell else s
        s = f"{_size_text(self.data.shape)} {self.kind}"
        return s if in_cell else f"[{s}]"

    def display_lines(self) -> list[str]:
        if self.data.size == 0:
            return []
        if self.data.ndim > 2:
            return [f"  {_size_text(self.data.shape)} {self.kind} array"]
        texts = self.element_texts()
        width = max(len(s) for s in texts.flat)
        return ["   " + "   ".join(str(s).ljust(width) for s in row).rstrip() for row in texts]

    def format_display(self, name: str) -> str:
        shape = self.data.shape
        if self.data.size == 0:
            return f"{name} = \n\n  {_size_text(shape)} empty {self.kind} array\n\n"
        head = f"  {self.kind}" if self.data.size == 1 else f"  {_size_text(shape)} {self.kind} array"
        return f"{name} = \n\n{head}\n\n" + "\n".join(self.display_lines()) + "\n\n"


class DatetimeArray(_TimeArray):
    kind = "datetime"

    def __init__(self, data: Any, fmt: str | None = None, zone: str = ""):
        super().__init__(data, fmt)
        self.zone = zone

    def like(self, data: Any) -> "DatetimeArray":
        return DatetimeArray(data, self.fmt, self.zone)

    @property
    def format(self) -> str:
        if self.fmt:
            return self.fmt
        d = self.data[np.isfinite(self.data)]
        return DEFAULT_DATE if d.size == 0 or bool(np.all(np.mod(d, DAY) == 0)) else DEFAULT_FULL

    def element_texts(self, fmt: str | None = None) -> np.ndarray:
        f = fmt or self.format
        out = np.empty(self.data.shape, dtype=object)
        for idx in np.ndindex(self.data.shape):
            out[idx] = format_datetime(float(self.data[idx]), f, self.zone)
        return out

    def coerce(self, value: Any) -> np.ndarray:
        if isinstance(value, DatetimeArray):
            return value.data
        items = _text_items(value)
        if items is not None:
            vals, _ = parse_texts(items[0], None)
            return vals.reshape(items[1], order="F")
        if isinstance(value, np.ndarray) and value.dtype.kind == "f" and (value.size == 0 or np.all(np.isnan(value))):
            return value.astype(float)
        raise MatlabError("Only datetime values, date text or NaT can be combined with a datetime array.",
                          "MATLAB:datetime:InvalidData")

    # properties
    _FIELDS = {"Year": 0, "Month": 1, "Day": 2, "Hour": 3, "Minute": 4, "Second": 5}

    def get_field(self, interp, name: str) -> Any:
        if name in self._FIELDS:
            return fields_from_seconds(self.data)[self._FIELDS[name]]
        if name == "Format":
            return char(self.format)
        if name == "TimeZone":
            return char(self.zone)
        if name == "SystemTimeZone":
            return char(time.tzname[0])
        raise MatlabError(f"Unrecognized property '{name}' for class 'datetime'.", "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> "DatetimeArray":
        if name in self._FIELDS:
            parts = list(fields_from_seconds(self.data))
            v = np.asarray(to_numeric(value), dtype=float)
            if v.size != 1 and v.shape != self.data.shape:
                raise MatlabError(f"The {name} values must be a scalar or match the size of the datetime array.")
            k = self._FIELDS[name]
            if k == 1:  # changing the month keeps the day, clamped to the month's length
                parts[1] = v + np.zeros(self.data.shape)
                y2 = parts[0] + np.floor((parts[1] - 1) / 12)
                m2 = np.mod(parts[1] - 1, 12) + 1
                parts[2] = np.minimum(parts[2], month_length(y2, m2))
            else:
                parts[k] = v + np.zeros(self.data.shape)
                if k == 0:
                    parts[2] = np.minimum(parts[2], month_length(parts[0], parts[1]))
            return self.like(np.where(np.isnan(self.data), np.nan, seconds_from_fields(*parts)))
        if name == "Format":
            f = text(value)
            if f.lower() in ("default", "defaultdate"):
                return DatetimeArray(self.data, DEFAULT_DATE if f.lower() == "defaultdate" else None, self.zone)
            _check_datetime_format(f)
            return DatetimeArray(self.data, f, self.zone)
        if name == "TimeZone":
            new = _zone(text(value))
            return DatetimeArray(_rezone(self.data, self.zone, new), self.fmt, new)
        raise MatlabError(f"Unrecognized or read-only property '{name}' for class 'datetime'.")

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        return _datetime_binop(op, left, right)

    def unary_op(self, interp, op: str) -> Any:
        if op in ("'", ".'"):
            return self.like(self.data.T) if self.data.ndim == 2 else NotImplemented
        return NotImplemented


def _rezone(data: np.ndarray, old: str, new: str) -> np.ndarray:
    if old == new or old == "" or new == "":
        return data
    off = local_offset()
    return data - off if (old, new) == ("local", "UTC") else data + off


def _check_datetime_format(f: str) -> None:
    for kind, _count in _tokens(f):
        if kind != "lit" and kind not in "yuMLdDeEHhmsSaQwZxXz":
            raise MatlabError(f"Unsupported letter '{kind}' in the datetime format '{f}'. Put literal text "
                              "in single quotes.", "MATLAB:datetime:InvalidFormat")


_UNIT_SECONDS = {"y": YEAR_SECONDS, "d": DAY, "h": 3600.0, "m": 60.0, "s": 1.0}
_UNIT_LABEL = {"y": ("yr", "yrs"), "d": ("day", "days"), "h": ("hr", "hr"), "m": ("min", "min"), "s": ("sec", "sec")}


def _num_text(x: float) -> str:
    if x == int(x) and abs(x) < 1e5:
        return str(int(x))
    return f"{x:.5g}"


def format_duration(s: float, fmt: str) -> str:
    if s != s:
        return f"NaN {_UNIT_LABEL[fmt][1]}" if fmt in _UNIT_SECONDS else "NaN"
    if fmt in _UNIT_SECONDS:
        x = s / _UNIT_SECONDS[fmt]
        if math.isinf(x):
            return ("Inf " if x > 0 else "-Inf ") + _UNIT_LABEL[fmt][1]
        return f"{_num_text(x)} {_UNIT_LABEL[fmt][0 if x == 1 else 1]}"
    if math.isinf(s):
        return "Inf" if s > 0 else "-Inf"
    parts = fmt.split(".")
    digits = len(parts[1]) if len(parts) > 1 else 0
    names = parts[0].split(":")
    sign = "-" if s < 0 else ""
    q = 10 ** digits
    total = math.floor(abs(s) * q + 0.5) / q if names[-1] == "ss" else abs(s)
    sizes = {"dd": DAY, "hh": 3600.0, "mm": 60.0, "ss": 1.0}
    out = []
    rem = total
    for k, nm in enumerate(names):
        unit = sizes[nm]
        if k == len(names) - 1:
            if nm == "ss":
                whole = math.floor(rem + 1e-9)
                frac = rem - whole
                txt = f"{int(whole):02d}"
                if digits:
                    txt += "." + f"{int(math.floor(max(frac, 0) * q + 0.5)):0{digits}d}"[:digits]
                out.append(txt)
            else:
                out.append(f"{int(math.floor(rem / unit + 1e-9)):02d}")
        else:
            whole = math.floor(rem / unit + 1e-9)
            rem -= whole * unit
            out.append(f"{int(whole):02d}")
    return sign + ":".join(out)


_TIMER_FORMATS = {"hh:mm:ss", "dd:hh:mm:ss", "mm:ss", "hh:mm"}


def _check_duration_format(f: str) -> str:
    base = f.split(".")[0]
    if f in _UNIT_SECONDS:
        return f
    if base in _TIMER_FORMATS and (("." not in f) or (base.endswith("ss") and set(f.split(".")[1]) == {"S"})):
        return f
    raise MatlabError(f"Unsupported duration format '{f}'. Use 'y', 'd', 'h', 'm', 's', 'hh:mm:ss', "
                      "'dd:hh:mm:ss', 'mm:ss' or 'hh:mm' (optionally with fractional seconds, 'hh:mm:ss.SSS').",
                      "MATLAB:duration:InvalidFormat")


class DurationArray(_TimeArray):
    kind = "duration"
    fill_value = 0.0

    def like(self, data: Any) -> "DurationArray":
        return DurationArray(data, self.fmt)

    @property
    def format(self) -> str:
        return self.fmt or "hh:mm:ss"

    def element_texts(self, fmt: str | None = None) -> np.ndarray:
        f = fmt or self.format
        out = np.empty(self.data.shape, dtype=object)
        for idx in np.ndindex(self.data.shape):
            out[idx] = format_duration(float(self.data[idx]), f)
        return out

    def coerce(self, value: Any) -> np.ndarray:
        if isinstance(value, DurationArray):
            return value.data
        if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            return value.astype(float) * DAY  # plain numbers count as days
        items = _text_items(value)
        if items is not None:
            return np.array([_parse_duration_text(s) for s in items[0]]).reshape(items[1], order="F")
        raise MatlabError("Only duration values or numbers (taken as days) can be combined with a duration array.")

    def get_field(self, interp, name: str) -> Any:
        if name == "Format":
            return char(self.format)
        raise MatlabError(f"Unrecognized property '{name}' for class 'duration'.", "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> "DurationArray":
        if name == "Format":
            return DurationArray(self.data, _check_duration_format(text(value)))
        raise MatlabError(f"Unrecognized or read-only property '{name}' for class 'duration'.")

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if isinstance(left, (DatetimeArray, CalendarDuration)) or isinstance(right, (DatetimeArray, CalendarDuration)):
            other = left if not isinstance(left, DurationArray) else right
            return other.binary_op(interp, op, left, right)
        return _duration_binop(op, left, right)

    def unary_op(self, interp, op: str) -> Any:
        if op == "-":
            return self.like(-self.data)
        if op == "+":
            return self
        if op in ("'", ".'") and self.data.ndim == 2:
            return self.like(self.data.T)
        return NotImplemented


def _parse_duration_text(s: str) -> float:
    m = re.match(r"\s*(-)?(?:(\d+):)?(\d+):(\d+)(?::(\d+(?:\.\d*)?))?\s*$", s)
    if not m:
        return float("nan")
    sign = -1.0 if m.group(1) else 1.0
    if m.group(2) is not None and m.group(5) is not None:  # dd:hh:mm:ss
        total = float(m.group(2)) * DAY + float(m.group(3)) * 3600 + float(m.group(4)) * 60 + float(m.group(5))
    elif m.group(2) is not None:  # hh:mm:ss (three fields captured as 2,3,4)
        total = float(m.group(2)) * 3600 + float(m.group(3)) * 60 + float(m.group(4))
    elif m.group(5) is not None:
        total = float(m.group(3)) * 3600 + float(m.group(4)) * 60 + float(m.group(5))
    else:  # hh:mm
        total = float(m.group(3)) * 3600 + float(m.group(4)) * 60
    return sign * total


class CalendarDuration(MObject):
    """Calendar lengths of time: whole months and days plus a time part (array-valued)."""

    def __init__(self, months: Any, days: Any, secs: Any):
        m, d, s = np.broadcast_arrays(*(np.atleast_2d(np.asarray(x, dtype=float)) for x in (months, days, secs)))
        self.months, self.days, self.secs = m.copy(), d.copy(), s.copy()

    def class_name(self) -> str:
        return "calendarDuration"

    def isa_class(self, name: str) -> bool:
        return name == "calendarDuration"

    def matlab_shape(self) -> tuple:
        return tuple(self.months.shape)

    def has_method(self, name: str) -> bool:
        return name in _NATIVE

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        res = _NATIVE[name](interp, args, nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, CalendarDuration) and self.months.shape == other.months.shape and bool(
            np.all(self.months == other.months) and np.all(self.days == other.days) and np.all(self.secs == other.secs))

    def index_get(self, interp, args: list) -> Any:
        return CalendarDuration(*(ix.get(a, list(args)) for a in (self.months, self.days, self.secs)))

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        parts = []
        for i in items:
            if isinstance(i, np.ndarray) and i.size == 0:
                continue
            if not isinstance(i, CalendarDuration):
                raise MatlabError("Calendar durations can only be concatenated with other calendar durations.")
            parts.append(i)
        return CalendarDuration(*(ops._concat_arrays([getattr(p, f) for p in parts], axis)
                                  for f in ("months", "days", "secs")))

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if isinstance(left, DatetimeArray) or isinstance(right, DatetimeArray):
            return _datetime_binop(op, left, right)
        if op in ("+", "-"):
            a, b = _as_calendar(left), _as_calendar(right)
            sg = 1.0 if op == "+" else -1.0
            return CalendarDuration(a.months + sg * b.months, a.days + sg * b.days, a.secs + sg * b.secs)
        if op in ("*", ".*"):
            cal, k = (left, right) if isinstance(left, CalendarDuration) else (right, left)
            if isinstance(k, np.ndarray) and k.dtype.kind in "biuf":
                kk = k.astype(float)
                if np.any(kk != np.round(kk)):
                    raise MatlabError("Calendar durations can only be multiplied by whole numbers.")
                return CalendarDuration(cal.months * kk, cal.days * kk, cal.secs * kk)
        if op in ("==", "~="):
            if isinstance(left, CalendarDuration) and isinstance(right, CalendarDuration):
                eq = (left.months == right.months) & (left.days == right.days) & (left.secs == right.secs)
                return eq if op == "==" else ~eq
        return NotImplemented

    def unary_op(self, interp, op: str) -> Any:
        if op == "-":
            return CalendarDuration(-self.months, -self.days, -self.secs)
        if op == "+":
            return self
        if op in ("'", ".'"):
            return CalendarDuration(self.months.T, self.days.T, self.secs.T)
        return NotImplemented

    def element_texts(self) -> np.ndarray:
        out = np.empty(self.months.shape, dtype=object)
        for idx in np.ndindex(self.months.shape):
            mo, d, s = float(self.months[idx]), float(self.days[idx]), float(self.secs[idx])
            if any(x != x for x in (mo, d, s)):
                out[idx] = "NaN"
                continue
            sign = -1 if mo < 0 else 1
            y, mm = divmod(abs(int(mo)), 12)
            parts = []
            if y:
                parts.append(f"{sign * y}y")
            if mm:
                parts.append(f"{sign * mm}mo")
            if d:
                parts.append(f"{int(d)}d")
            if s:
                sg = "-" if s < 0 else ""
                a = abs(s)
                h, rem = divmod(a, 3600)
                mi, sec = divmod(rem, 60)
                parts.append(f"{sg}{int(h)}h {int(mi)}m {_num_text(sec)}s")
            out[idx] = " ".join(parts) if parts else "0d"
        return out

    def summary_text(self, in_cell: bool) -> str:
        if self.months.size == 1:
            return str(self.element_texts().flat[0])
        s = f"{_size_text(self.months.shape)} calendarDuration"
        return s if in_cell else f"[{s}]"

    def display_lines(self) -> list[str]:
        if self.months.size == 0:
            return []
        texts = self.element_texts()
        width = max(len(s) for s in texts.flat)
        return ["   " + "   ".join(str(s).rjust(width) for s in row) for row in texts]

    def format_display(self, name: str) -> str:
        n = self.months.size
        head = "  calendarDuration" if n == 1 else f"  {_size_text(self.months.shape)} calendarDuration array"
        return f"{name} = \n\n{head}\n\n" + "\n".join(self.display_lines()) + "\n\n"


def _as_calendar(v: Any) -> CalendarDuration:
    if isinstance(v, CalendarDuration):
        return v
    if isinstance(v, DurationArray):
        return CalendarDuration(0.0, 0.0, v.data)
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return CalendarDuration(0.0, 0.0, v.astype(float) * DAY)
    raise MatlabError("Calendar durations can only be combined with durations, datetimes or numbers of days.")


# -- operators ----------------------------------------------------------------------------------------------

def _bc(a: np.ndarray, b: np.ndarray, op: str) -> tuple[np.ndarray, np.ndarray]:
    a, b = ops.align(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    ops._check_broadcast(a, b, op)
    return a, b


_CMP = {"==": np.equal, "~=": np.not_equal, "<": np.less, "<=": np.less_equal, ">": np.greater,
        ">=": np.greater_equal}


def _compare(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, y = _bc(a, b, op)
    with np.errstate(invalid="ignore"):
        return as_2d(np.asarray(_CMP[op](x, y)))


def _other_datetime(v: Any, like: DatetimeArray) -> np.ndarray:
    if isinstance(v, DatetimeArray):
        if (v.zone == "") != (like.zone == ""):
            raise MatlabError("Cannot combine or compare a datetime array with a time zone with one without "
                              "a time zone.", "MATLAB:datetime:IncompatibleTZ")
        return _rezone(v.data, v.zone, like.zone)
    items = _text_items(v)
    if items is not None:
        vals, _ = parse_texts(items[0], None)
        return vals.reshape(items[1], order="F")
    raise MatlabError("Comparison is not defined between datetime arrays and "
                      f"{getattr(v, 'class_name', lambda: 'numeric')()} arrays.", "MATLAB:datetime:InvalidComparison")


def _apply_calendar(data: np.ndarray, cal: CalendarDuration, sign: float, op: str) -> np.ndarray:
    d, mo = _bc(data, cal.months, op)
    out = add_months(d, sign * mo) if np.any(cal.months != 0) else d + 0 * mo
    out, days = _bc(out, cal.days, op)
    out = out + sign * days * DAY
    out, secs = _bc(out, cal.secs, op)
    return as_2d(np.asarray(out + sign * secs))


def _datetime_binop(op: str, left: Any, right: Any) -> Any:
    ld, rd = isinstance(left, DatetimeArray), isinstance(right, DatetimeArray)
    if op in _CMP:
        ref = left if ld else right
        a = left.data if ld else _other_datetime(left, ref)
        b = _other_datetime(right, ref) if ld else right.data
        return _compare(op, a, b)
    if op == "+" or op == "-":
        if ld and rd:
            if op == "+":
                raise MatlabError("Addition is not defined between datetime arrays.",
                                  "MATLAB:datetime:DatetimeAdditionNotDefined")
            a, b = _bc(left.data, _other_datetime(right, left), op)
            return DurationArray(as_2d(np.asarray(a - b)), "hh:mm:ss")
        if rd and op == "-":
            if _text_items(left) is not None:
                a, b = _bc(_other_datetime(left, right), right.data, op)
                return DurationArray(as_2d(np.asarray(a - b)), "hh:mm:ss")
            raise MatlabError("Subtracting a datetime from a duration or number is not defined.")
        dt, other = (left, right) if ld else (right, left)
        sign = 1.0 if op == "+" else -1.0
        if isinstance(other, CalendarDuration):
            return dt.like(_apply_calendar(dt.data, other, sign, op))
        if isinstance(other, DurationArray):
            delta = other.data
        elif isinstance(other, np.ndarray) and other.dtype.kind in "biuf":
            delta = other.astype(float) * DAY  # numbers count as days
        elif op == "-" and _text_items(other) is not None:
            a, b = _bc(dt.data, _other_datetime(other, dt), op)
            return DurationArray(as_2d(np.asarray(a - b)), "hh:mm:ss")
        else:
            raise MatlabError(f"Operator '{op}' is not defined between a datetime array and a "
                              f"{getattr(other, 'class_name', lambda: type(other).__name__)()} value.")
        a, b = _bc(dt.data, delta, op)
        return dt.like(as_2d(np.asarray(a + sign * b)))
    return NotImplemented


def _dur_seconds(v: Any, numbers_are_days: bool = True) -> np.ndarray | None:
    if isinstance(v, DurationArray):
        return v.data
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return v.astype(float) * (DAY if numbers_are_days else 1.0)
    return None


def _duration_binop(op: str, left: Any, right: Any) -> Any:
    ld, rd = isinstance(left, DurationArray), isinstance(right, DurationArray)
    ref = left if ld else right
    if op in _CMP or op in ("+", "-"):
        a, b = _dur_seconds(left), _dur_seconds(right)
        if a is None or b is None:
            if op in _CMP:
                items = _text_items(right if ld else left)
                if items is not None:
                    other = np.array([_parse_duration_text(s) for s in items[0]]).reshape(items[1], order="F")
                    return _compare(op, left.data if ld else other, other if ld else right.data)
            return NotImplemented
        if op in _CMP:
            return _compare(op, a, b)
        x, y = _bc(a, b, op)
        return ref.like(as_2d(np.asarray(x + y if op == "+" else x - y)))
    if op in ("*", ".*"):
        if ld and rd:
            raise MatlabError("Multiplication is not defined between two duration arrays.")
        k = right if ld else left
        if not (isinstance(k, np.ndarray) and k.dtype.kind in "biuf"):
            return NotImplemented
        if op == "*" and ref.data.size != 1 and k.size != 1:
            raise MatlabError("Use .* to multiply a duration array by a numeric array element by element.")
        x, y = _bc(ref.data, k.astype(float), op)
        return ref.like(as_2d(np.asarray(x * y)))
    if op in ("/", "./"):
        if ld and rd:
            x, y = _bc(left.data, right.data, op)
            with np.errstate(divide="ignore", invalid="ignore"):
                return as_2d(np.asarray(x / y))
        if ld and isinstance(right, np.ndarray) and right.dtype.kind in "biuf":
            x, y = _bc(left.data, right.astype(float), op)
            with np.errstate(divide="ignore", invalid="ignore"):
                return left.like(as_2d(np.asarray(x / y)))
        raise MatlabError("A number cannot be divided by a duration.")
    return NotImplemented


# -- constructors ---------------------------------------------------------------------------------------------

def _split_options(args: list, names: tuple) -> tuple[list, dict]:
    """Peel trailing name/value pairs off an argument list (names are case-insensitive)."""
    args = list(args)
    opts: dict = {}
    lookup = {n.lower(): n for n in names}
    while len(args) >= 2 and is_text(args[-2]) and text(args[-2]).lower() in lookup:
        opts[lookup[text(args[-2]).lower()]] = args[-1]
        del args[-2:]
    return args, opts


def _numeric_arg(v: Any, what: str) -> np.ndarray:
    if not (isinstance(v, np.ndarray) and v.dtype.kind in "biuf"):
        raise MatlabError(f"{what} must be numeric.")
    return v.astype(float)


def _broadcast_parts(parts: list, name: str) -> list:
    shape = None
    for p in parts:
        if p.size != 1:
            if shape is not None and p.shape != shape:
                raise MatlabError(f"{name}: the numeric inputs must be scalars or arrays of the same size.")
            shape = p.shape
    shape = shape or (1, 1)
    return [np.broadcast_to(p, shape) if p.size == 1 else p for p in parts]


_DATETIME_OPTS = ("Format", "InputFormat", "ConvertFrom", "TimeZone", "Locale", "PivotYear", "Epoch", "TicksPerSecond")


@builtin("datetime")
def _datetime(interp, args, nargout):
    args, opts = _split_options(args, _DATETIME_OPTS)
    zone = _zone(text(opts["TimeZone"])) if "TimeZone" in opts else ""
    fmt = None
    if "Format" in opts:
        f = text(opts["Format"])
        if f.lower() not in ("default", "preserveinput"):
            _check_datetime_format(f)
            fmt = f
    if not args:
        return DatetimeArray(np.array([[now_seconds(zone)]]), fmt or DEFAULT_FULL, zone)
    first = args[0]
    if isinstance(first, DatetimeArray):
        out = DatetimeArray(first.data, fmt or first.fmt, first.zone)
        return out.set_field(interp, "TimeZone", char(zone)) if "TimeZone" in opts else out
    if "ConvertFrom" in opts:
        kind = text(opts["ConvertFrom"]).lower()
        v = _numeric_arg(first, "The value to convert")
        if kind == "datenum":
            secs, shown = (v - EPOCH_DATENUM) * DAY, DEFAULT_FULL
            secs = np.round(secs * 1e6) / 1e6
        elif kind == "posixtime":
            secs, shown = v + (local_offset() if zone == "local" else 0.0), DEFAULT_FULL
        elif kind == "excel":
            secs, shown = (v - 25569.0) * DAY, DEFAULT_FULL
            secs = np.round(secs * 1e6) / 1e6
        elif kind == "excel1904":
            secs, shown = np.round((v - 24107.0) * DAY * 1e6) / 1e6, DEFAULT_FULL
        elif kind == "juliandate":
            secs, shown = np.round((v - 2440587.5) * DAY * 1e3) / 1e3, DEFAULT_FULL
        elif kind == "modifiedjuliandate":
            secs, shown = np.round((v - 40587.0) * DAY * 1e3) / 1e3, DEFAULT_FULL
        elif kind == "yyyymmdd":
            secs = seconds_from_fields(np.floor(v / 10000), np.floor(np.mod(v, 10000) / 100), np.mod(v, 100))
            shown = DEFAULT_DATE
        elif kind == "epochtime":
            epoch = _datetime(interp, [opts["Epoch"]], 1).data.flat[0] if "Epoch" in opts else 0.0
            ticks = float(to_numeric(opts["TicksPerSecond"]).flat[0]) if "TicksPerSecond" in opts else 1.0
            secs, shown = epoch + v / ticks, DEFAULT_FULL
        else:
            raise MatlabError(f"Unknown ConvertFrom value '{text(opts['ConvertFrom'])}'. Use 'datenum', 'posixtime', "
                              "'excel', 'juliandate', 'modifiedjuliandate', 'yyyymmdd' or 'epochtime'.")
        if fmt is None and np.all(np.mod(secs[np.isfinite(secs)], DAY) == 0):
            shown = DEFAULT_DATE
        return DatetimeArray(secs, fmt or shown, zone)
    if is_text(first) and text(first).lower() in ("now", "today", "tomorrow", "yesterday") and len(args) == 1:
        word = text(first).lower()
        t = now_seconds(zone)
        if word != "now":
            t = math.floor(t / DAY) * DAY + {"today": 0.0, "tomorrow": DAY, "yesterday": -DAY}[word]
        return DatetimeArray(np.array([[t]]), fmt or (DEFAULT_FULL if word == "now" else DEFAULT_DATE), zone)
    items = _text_items(first)
    if items is not None:
        if len(args) > 1:
            raise MatlabError("datetime: unexpected extra inputs after the date text. Options are name-value "
                              "pairs such as 'InputFormat' and 'Format'.")
        infmt = text(opts["InputFormat"]) if "InputFormat" in opts else None
        vals, has_time = parse_texts(items[0], infmt)
        shown = fmt or (DEFAULT_FULL if has_time else DEFAULT_DATE)
        return DatetimeArray(vals.reshape(items[1], order="F"), shown, zone)
    if isinstance(first, Cell):
        raise MatlabError("datetime: a cell array input must contain only text.")
    nums = [_numeric_arg(a, "Date and time components") for a in args]
    if len(nums) == 1:
        v = nums[0]
        if v.ndim == 2 and v.shape[1] in (3, 6) and v.size:
            nums = [v[:, k:k + 1] for k in range(v.shape[1])]
        elif v.size == 0:
            return DatetimeArray(np.zeros((0, 0)), fmt, zone)
        else:
            raise MatlabError("datetime: numeric input must be Y,M,D or Y,M,D,H,MI,S values (as separate inputs or "
                              "as the columns of a matrix), or be used with 'ConvertFrom'.")
    if len(nums) not in (3, 6, 7):
        raise MatlabError("datetime: give 3 (Y,M,D), 6 (Y,M,D,H,MI,S) or 7 (..., MS) numeric inputs.")
    parts = _broadcast_parts(nums, "datetime")
    secs = seconds_from_fields(*parts[:6]) if len(parts) >= 6 else seconds_from_fields(*parts[:3])
    if len(parts) == 7:
        secs = secs + parts[6] / 1000.0
    return DatetimeArray(secs, fmt or (DEFAULT_DATE if len(parts) == 3 else DEFAULT_FULL), zone)


def _size_from(args: list) -> tuple:
    if not args:
        return (1, 1)
    if len(args) == 1:
        v = to_numeric(args[0]).flatten()
        return (int(v[0]), int(v[0])) if v.size == 1 else tuple(int(x) for x in v)
    return tuple(int(to_numeric(a).flat[0]) for a in args)


@builtin("NaT")
def _nat(interp, args, nargout):
    args, opts = _split_options(args, ("Format", "TimeZone"))
    zone = _zone(text(opts["TimeZone"])) if "TimeZone" in opts else ""
    out = DatetimeArray(np.full(_size_from(args), np.nan), None, zone)
    return out.set_field(interp, "Format", opts["Format"]) if "Format" in opts else out


@builtin("datetime.empty")
def _datetime_empty(interp, args, nargout):
    return DatetimeArray(np.zeros(_size_from(args) if args else (0, 0)))


@builtin("duration.empty")
def _duration_empty(interp, args, nargout):
    return DurationArray(np.zeros(_size_from(args) if args else (0, 0)))


@builtin("duration")
def _duration(interp, args, nargout):
    args, opts = _split_options(args, ("Format", "InputFormat"))
    fmt = _check_duration_format(text(opts["Format"])) if "Format" in opts else "hh:mm:ss"
    if not args:
        return DurationArray(np.zeros((1, 1)), fmt)
    first = args[0]
    if isinstance(first, DurationArray):
        return DurationArray(first.data, fmt if "Format" in opts else first.fmt)
    items = _text_items(first)
    if items is not None:
        vals = np.array([_parse_duration_text(s) for s in items[0]])
        if np.any(np.isnan(vals)):
            raise MatlabError("duration: could not read the text as a duration. Use the form 'hh:mm:ss' or "
                              "'dd:hh:mm:ss'.")
        return DurationArray(vals.reshape(items[1], order="F"), fmt)
    nums = [_numeric_arg(a, "Duration components") for a in args]
    if len(nums) == 1 and nums[0].ndim == 2 and nums[0].shape[1] == 3:
        nums = [nums[0][:, k:k + 1] for k in range(3)]
    if len(nums) not in (3, 4):
        raise MatlabError("duration: give hours, minutes and seconds (and optionally milliseconds).")
    parts = _broadcast_parts(nums, "duration")
    secs = parts[0] * 3600 + parts[1] * 60 + parts[2] + (parts[3] / 1000.0 if len(parts) == 4 else 0.0)
    return DurationArray(secs, fmt)


def _unit_function(name: str, unit: float, fmt: str):
    def f(interp, args, nargout):
        if not args:
            return DurationArray(np.array([[unit]]), fmt)
        v = args[0]
        if isinstance(v, DurationArray):
            return v.data / unit
        if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
            return DurationArray(v.astype(float) * unit, fmt)
        raise MatlabError(f"{name}: the input must be numeric or a duration array.")
    builtin(name)(f)
    _NATIVE[name] = f


_unit_function("seconds", 1.0, "s")
_unit_function("minutes", 60.0, "m")
_unit_function("hours", 3600.0, "h")
_unit_function("days", DAY, "d")
_unit_function("years", YEAR_SECONDS, "y")
_unit_function("milliseconds", 1e-3, "s")


def _calendar_function(name: str, months: float, days: float):
    def f(interp, args, nargout):
        v = _numeric_arg(args[0], f"The input to {name}") if args else np.array([[1.0]])
        if np.any(v != np.round(v)):
            raise MatlabError(f"{name}: the input must contain whole numbers.")
        return CalendarDuration(v * months, v * days, 0.0)
    builtin(name)(f)


_calendar_function("calyears", 12, 0)
_calendar_function("calquarters", 3, 0)
_calendar_function("calmonths", 1, 0)
_calendar_function("calweeks", 0, 7)
_calendar_function("caldays", 0, 1)


@builtin("isdatetime")
def _isdatetime(interp, args, nargout):
    return bool_scalar(isinstance(args[0], DatetimeArray))


@builtin("isduration")
def _isduration(interp, args, nargout):
    return bool_scalar(isinstance(args[0], DurationArray))


@builtin("iscalendarduration")
def _iscalendarduration(interp, args, nargout):
    return bool_scalar(isinstance(args[0], CalendarDuration))


# -- generic array functions -----------------------------------------------------------------------------------

def _first_time(args: list) -> _TimeArray:
    return next(a for a in args if isinstance(a, _TimeArray))


def _raw_args(args: list, ref: _TimeArray) -> list:
    out = []
    for a in args:
        if isinstance(a, _TimeArray):
            if type(a) is not type(ref):
                raise MatlabError(f"Cannot combine {ref.kind} and {a.kind} arrays in this function.")
            out.append(a.data)
        elif is_text(a) and text(a).lower() in ("omitnat", "includenat", "omitmissing", "includemissing"):
            out.append(char(text(a).lower().replace("nat", "nan").replace("missing", "nan")))
        else:
            out.append(a)
    return out


def _rewrap(name: str, wrap_outputs: int = 1, as_duration: bool = False):
    """Run the numeric built-in on the underlying seconds and wrap the first output(s) again."""
    def f(interp, args, nargout):
        ref = _first_time(args)
        res = interp.call_named(name, _raw_args(args, ref), nargout, interp.frame)
        out = []
        for k, r in enumerate(res):
            if k < wrap_outputs and isinstance(r, np.ndarray):
                out.append(DurationArray(r, "hh:mm:ss" if not isinstance(ref, DurationArray) else ref.fmt)
                           if as_duration else ref.like(r))
            else:
                out.append(r)
        return tuple(out) if len(out) != 1 else out[0]
    return f


for _n in ("sort", "unique", "min", "max", "mean", "median", "mode", "reshape", "repmat", "permute", "flip",
           "fliplr", "flipud", "circshift", "squeeze", "rot90", "cummax", "cummin", "linspace", "transpose",
           "ctranspose", "sortrows", "topkrows", "union", "intersect", "setdiff", "rmmissing", "bounds"):
    _NATIVE[_n] = _rewrap(_n, 2 if _n == "bounds" else 1)
for _n in ("ismember", "issorted", "numel", "isequaln", "issortedrows", "nnz"):
    _NATIVE[_n] = _rewrap(_n, 0)


def _by_kind(name: str):
    """diff / range / std: datetimes give durations, durations stay durations."""
    wrap = _rewrap(name, 1, as_duration=True)
    keep = _rewrap(name, 1)

    def f(interp, args, nargout):
        return (keep if isinstance(_first_time(args), DurationArray) else wrap)(interp, args, nargout)
    return f


for _n in ("diff", "range", "std"):
    _NATIVE[_n] = _by_kind(_n)


def _duration_only(name: str, wrap: int = 1):
    inner = _rewrap(name, wrap)

    def f(interp, args, nargout):
        if not isinstance(_first_time(args), DurationArray):
            raise MatlabError(f"Function '{name}' is not defined for datetime arrays.")
        return inner(interp, args, nargout)
    return f


for _n in ("sum", "cumsum", "abs", "mod", "rem"):
    _NATIVE[_n] = _duration_only(_n)
_NATIVE["sign"] = _duration_only("sign", 0)


def _rounder(name: str):
    def f(interp, args, nargout):
        d = args[0]
        if not isinstance(d, DurationArray):
            raise MatlabError(f"Use dateshift to round datetime values; '{name}' works on durations.")
        unit = 1.0
        if len(args) > 1:
            key = text(args[1]).lower().rstrip("s")
            units = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": DAY, "year": YEAR_SECONDS}
            if key not in units:
                raise MatlabError(f"{name}: unknown unit '{text(args[1])}'.")
            unit = units[key]
        res = interp.call_named(name, [d.data / unit], 1, interp.frame)[0]
        return d.like(res * unit)
    return f


for _n in ("floor", "ceil", "round", "fix"):
    _NATIVE[_n] = _rounder(_n)


@native("isnat")
def _isnat(interp, args, nargout):
    v = args[0]
    if not isinstance(v, DatetimeArray):
        raise MatlabError("isnat: the input must be a datetime array.")
    return np.isnan(v.data)


builtin("isnat")(_isnat)


@native("isnan", "ismissing")
def _isnan(interp, args, nargout):
    v = args[0]
    if isinstance(v, CalendarDuration):
        return np.isnan(v.months) | np.isnan(v.days) | np.isnan(v.secs)
    return np.isnan(v.data)


@native("isfinite")
def _isfinite(interp, args, nargout):
    return np.isfinite(args[0].data)


@native("isinf")
def _isinf(interp, args, nargout):
    return np.isinf(args[0].data)


@native("isequal")
def _isequal(interp, args, nargout):
    first = args[0]
    return bool_scalar(all(hasattr(first, "object_equals") and first.object_equals(b) for b in args[1:]))


def _op_function(op: str):
    def f(interp, args, nargout):
        from ..classdef import call_operator
        return call_operator(interp, op, args[0], args[1])
    return f


for _op, _fname in {"+": "plus", "-": "minus", ".*": "times", "*": "mtimes", "./": "rdivide", "/": "mrdivide",
                    "==": "eq", "~=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}.items():
    _NATIVE[_fname] = _op_function(_op)


@native("uminus")
def _uminus(interp, args, nargout):
    res = args[0].unary_op(interp, "-")
    if res is NotImplemented:
        raise MatlabError("Unary minus is not defined for datetime arrays.")
    return res


def _cat(axis: int | None):
    def f(interp, args, nargout):
        items = list(args)
        ax = axis
        if ax is None:
            ax = int(to_numeric(items.pop(0)).flat[0]) - 1
        return ops.concat(items, 1 if ax == 1 else 0) if ax in (0, 1) else _nd_cat(items, ax)
    return f


def _nd_cat(items: list, ax: int):
    ref = _first_time(items)
    datas = []
    for i in items:
        d = i.data if isinstance(i, _TimeArray) else ref.coerce(i)
        datas.append(d.reshape(d.shape + (1,) * (ax + 1 - d.ndim)))
    return ref.like(np.concatenate(datas, axis=ax))


_NATIVE["cat"] = _cat(None)
_NATIVE["horzcat"] = _cat(1)
_NATIVE["vertcat"] = _cat(0)


def colon_range(start: Any, step: Any, stop: Any) -> Any:
    """``start:step:stop`` for datetime and duration endpoints (step defaults to one day)."""
    ref = start if isinstance(start, _TimeArray) else stop
    if not isinstance(ref, _TimeArray) or ref.data.size == 0:
        raise MatlabError("colon: the endpoints must be datetime or duration scalars.")

    def scalar(v: Any) -> float:
        d = v.data if isinstance(v, _TimeArray) else ref.coerce(v)
        if d.size != 1:
            raise MatlabError("The endpoints of a datetime or duration range must be scalars.")
        return float(d.flat[0])

    if isinstance(start, _TimeArray) and isinstance(stop, _TimeArray) and type(start) is not type(stop):
        raise MatlabError("colon: both endpoints must be of the same type.")
    a, b = scalar(start), scalar(stop)
    if isinstance(step, CalendarDuration):
        if not isinstance(ref, DatetimeArray) or step.months.size != 1:
            raise MatlabError("colon: a calendar step needs datetime endpoints and a scalar step.")
        forward = (float(step.months.flat[0]) * 30 + float(step.days.flat[0])) * DAY + float(step.secs.flat[0])
        vals = []
        if forward != 0 and a == a and b == b:
            est = int(abs(b - a) / abs(forward) * 1.2) + 3
            k = np.arange(0, est, dtype=float).reshape(1, -1)
            cand = _apply_calendar(np.array([[a]]), CalendarDuration(step.months * k, step.days * k, step.secs * k),
                                   1.0, "+").ravel()
            vals = cand[cand <= b] if forward > 0 else cand[cand >= b]
        return ref.like(np.asarray(vals, dtype=float).reshape(1, -1))
    if step is None:
        inc = DAY
    elif isinstance(step, DurationArray):
        inc = float(step.data.flat[0])
    elif isinstance(step, np.ndarray) and step.dtype.kind in "biuf" and step.size == 1:
        inc = float(step.flat[0]) * DAY
    else:
        raise MatlabError("colon: the step must be a duration, a calendar duration or a number of days.")
    if inc == 0 or a != a or b != b or (inc > 0 and a > b) or (inc < 0 and a < b):
        return ref.like(np.zeros((1, 0)))
    n = int(math.floor((b - a) / inc + 1e-10)) + 1
    return ref.like((a + inc * np.arange(n, dtype=float)).reshape(1, -1))


@native("colon")
def _colon(interp, args, nargout):
    if len(args) == 2:
        return colon_range(args[0], None, args[1])
    if len(args) == 3:
        return colon_range(args[0], args[1], args[2])
    raise MatlabError("colon expects two or three inputs.")


# -- datetime functions ---------------------------------------------------------------------------------------------

def _as_seconds(v: Any, what: str = "input") -> tuple[np.ndarray, DatetimeArray | None]:
    """Seconds for a datetime, date text or serial date numbers."""
    if isinstance(v, DatetimeArray):
        return v.data, v
    items = _text_items(v)
    if items is not None:
        vals, _ = parse_texts(items[0], None)
        return vals.reshape(items[1], order="F"), None
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return np.round((v.astype(float) - EPOCH_DATENUM) * DAY * 1e6) / 1e6, None
    raise MatlabError(f"The {what} must be a datetime array, date text or serial date numbers.")


def _component(index: int):
    def f(interp, args, nargout):
        secs, _ = _as_seconds(args[0])
        return fields_from_seconds(secs)[index]
    return f


for _i, _n in ((3, "hour"), (4, "minute"), (5, "second")):
    builtin(_n)(_component(_i))
    _NATIVE[_n] = _component(_i)


def _names_cell(idx: np.ndarray, names: list, short: bool) -> Any:
    out = np.empty(idx.shape, dtype=object)
    for k in np.ndindex(idx.shape):
        v = idx[k]
        nm = "" if v != v else names[int(v)]
        out[k] = char(nm[:3] if short else nm)
    return Cell(out)


def _day_of_year(secs: np.ndarray) -> np.ndarray:
    y = fields_from_seconds(secs)[0]
    with np.errstate(invalid="ignore"):
        return np.floor(secs / DAY) - days_from_civil(y, 1, 1) + 1


def _week_of_year(secs: np.ndarray) -> np.ndarray:
    """Week number where week 1 contains January 1 and weeks start on Sunday."""
    y = fields_from_seconds(secs)[0]
    jan1 = days_from_civil(y, 1, 1)
    with np.errstate(invalid="ignore"):
        return np.floor((_day_of_year(secs) - 1 + np.mod(jan1 + 4, 7)) / 7) + 1


@native("month")
def _month(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    m = fields_from_seconds(secs)[1]
    kind = text(args[1]).lower() if len(args) > 1 else "monthofyear"
    if kind in ("name", "shortname"):
        return _names_cell(m - 1, MONTHS, kind == "shortname")
    if kind != "monthofyear":
        raise MatlabError(f"month: unknown option '{text(args[1])}'. Use 'monthofyear', 'name' or 'shortname'.")
    return m


builtin("month")(_month)


@native("day")
def _day(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    kind = text(args[1]).lower() if len(args) > 1 else "dayofmonth"
    if kind == "dayofmonth":
        return fields_from_seconds(secs)[2]
    if kind == "dayofyear":
        return _day_of_year(secs)
    if kind == "dayofweek":
        return weekday_index(secs) + 1
    if kind in ("name", "shortname"):
        return _names_cell(weekday_index(secs), WEEKDAYS, kind == "shortname")
    raise MatlabError(f"day: unknown option '{text(args[1])}'. Use 'dayofmonth', 'dayofweek', 'dayofyear', "
                      "'name' or 'shortname'.")


builtin("day")(_day)


@native("year")
def _year(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    return fields_from_seconds(secs)[0]


builtin("year")(_year)


@native("quarter")
def _quarter(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    return np.floor((fields_from_seconds(secs)[1] - 1) / 3) + 1


builtin("quarter")(_quarter)


@native("week")
def _week(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    kind = text(args[1]).lower() if len(args) > 1 else "weekofyear"
    if kind == "weekofyear":
        return _week_of_year(secs)
    if kind == "weekofmonth":
        y, m, d = fields_from_seconds(secs)[:3]
        first = np.mod(days_from_civil(y, m, 1) + 4, 7)
        return np.floor((d - 1 + first) / 7) + 1
    raise MatlabError(f"week: unknown option '{text(args[1])}'. Use 'weekofyear' or 'weekofmonth'.")


builtin("week")(_week)


@native("weekday")
def _weekday(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    n = weekday_index(secs) + 1
    if nargout < 2:
        return n
    long = any(is_text(a) and text(a).lower() == "long" for a in args[1:])
    names = [("" if v != v else (WEEKDAYS[int(v) - 1] if long else WEEKDAYS[int(v) - 1][:3]))
             for v in n.flatten(order="F")]
    width = max((len(s) for s in names), default=0)
    rows = np.array([list(s.ljust(width)) for s in names], dtype="<U1").reshape(len(names), width)
    return n, rows


builtin("weekday")(_weekday)


@native("ymd")
def _ymd(interp, args, nargout):
    y, m, d = fields_from_seconds(_as_seconds(args[0])[0])[:3]
    return (y, m, d)[:max(nargout, 1)]


builtin("ymd")(_ymd)


@native("hms")
def _hms(interp, args, nargout):
    v = args[0]
    if isinstance(v, DurationArray):
        a = np.abs(v.data)
        sg = np.sign(v.data)
        h = np.floor(a / 3600)
        mi = np.floor((a - h * 3600) / 60)
        return (sg * h, sg * mi, sg * (a - h * 3600 - mi * 60))[:max(nargout, 1)]
    h, mi, s = fields_from_seconds(_as_seconds(v)[0])[3:]
    return (h, mi, s)[:max(nargout, 1)]


builtin("hms")(_hms)


@native("timeofday")
def _timeofday(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    with np.errstate(invalid="ignore"):
        return DurationArray(np.mod(secs, DAY), "hh:mm:ss")


builtin("timeofday")(_timeofday)


@native("yyyymmdd")
def _yyyymmdd(interp, args, nargout):
    y, m, d = fields_from_seconds(_as_seconds(args[0])[0])[:3]
    return y * 10000 + m * 100 + d


builtin("yyyymmdd")(_yyyymmdd)


@native("posixtime")
def _posixtime(interp, args, nargout):
    v = args[0]
    if not isinstance(v, DatetimeArray):
        raise MatlabError("posixtime: the input must be a datetime array.")
    return v.data - (local_offset() if v.zone == "local" else 0.0)


builtin("posixtime")(_posixtime)


@native("juliandate")
def _juliandate(interp, args, nargout):
    secs, _ = _as_seconds(args[0])
    if len(args) > 1 and text(args[1]).lower() == "modifiedjuliandate":
        return secs / DAY + 40587.0
    return secs / DAY + 2440587.5


builtin("juliandate")(_juliandate)


@native("exceltime")
def _exceltime(interp, args, nargout):
    return _as_seconds(args[0])[0] / DAY + 25569.0


builtin("exceltime")(_exceltime)


@native("convertTo")
def _convert_to(interp, args, nargout):
    kind = text(args[1]).lower()
    table = {"posixtime": _posixtime, "juliandate": _juliandate, "excel": _exceltime, "datenum": _datenum,
             "yyyymmdd": _yyyymmdd}
    if kind == "modifiedjuliandate":
        return _juliandate(interp, [args[0], char("modifiedjuliandate")], 1)
    if kind not in table:
        raise MatlabError(f"convertTo: unknown target '{text(args[1])}'.")
    return table[kind](interp, [args[0]], 1)


builtin("convertTo")(_convert_to)


def _texts_of(v: Any, fmt_arg: Any = None) -> np.ndarray:
    if isinstance(v, CalendarDuration):
        return v.element_texts()
    if fmt_arg is not None:
        f = text(fmt_arg)
        if isinstance(v, DatetimeArray):
            _check_datetime_format(f)
        else:
            _check_duration_format(f)
        return v.element_texts(f)
    return v.element_texts()


@native("string")
def _string(interp, args, nargout):
    return string_result(_texts_of(args[0], args[1] if len(args) > 1 else None))


@native("cellstr")
def _cellstr(interp, args, nargout):
    texts = _texts_of(args[0], args[1] if len(args) > 1 else None)
    out = np.empty(texts.shape, dtype=object)
    for idx in np.ndindex(texts.shape):
        out[idx] = char(texts[idx])
    return Cell(out)


@native("char")
def _char(interp, args, nargout):
    texts = list(_texts_of(args[0], args[1] if len(args) > 1 else None).flatten(order="F"))
    return _char_rows(texts)


def _char_rows(rows: list) -> np.ndarray:
    if not rows:
        return char("")
    width = max(len(r) for r in rows)
    if width == 0:
        return np.zeros((len(rows), 0), dtype="<U1") if len(rows) != 1 else char("")
    return np.array([list(r.ljust(width)) for r in rows], dtype="<U1").reshape(len(rows), width)


@native("num2str", "disp_text")
def _num2str(interp, args, nargout):
    return _char(interp, args[:1], nargout)


@native("isbetween")
def _isbetween(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("isbetween expects isbetween(t, lower, upper).")
    t = args[0]
    kind = text(args[3]).lower() if len(args) > 3 else "closed"
    lo_op, hi_op = {"closed": (">=", "<="), "open": (">", "<"), "openleft": (">", "<="), "closedright": (">", "<="),
                    "openright": (">=", "<"), "closedleft": (">=", "<")}.get(kind, (None, None))
    if lo_op is None:
        raise MatlabError(f"isbetween: unknown interval type '{text(args[3])}'.")
    from ..classdef import call_operator
    a = call_operator(interp, lo_op, t, args[1])
    b = call_operator(interp, hi_op, t, args[2])
    return ops.binop("&", a, b)


builtin("isbetween")(_isbetween)


def _unit_start(secs: np.ndarray, unit: str, shift: float = 0.0) -> np.ndarray:
    """Start of the unit containing each time, moved by ``shift`` whole units."""
    y, m, d, h, mi, s = fields_from_seconds(secs)
    if unit == "year":
        return seconds_from_fields(y + shift, 1, 1)
    if unit == "quarter":
        return seconds_from_fields(y, np.floor((m - 1) / 3) * 3 + 1 + 3 * shift, 1)
    if unit == "month":
        return seconds_from_fields(y, m + shift, 1)
    if unit == "week":
        with np.errstate(invalid="ignore"):
            day0 = np.floor(secs / DAY)
            return (day0 - np.mod(day0 + 4, 7) + 7 * shift) * DAY
    size = {"day": DAY, "hour": 3600.0, "minute": 60.0, "second": 1.0}[unit]
    with np.errstate(invalid="ignore"):
        return (np.floor(secs / size) + shift) * size


@native("dateshift")
def _dateshift(interp, args, nargout):
    if len(args) < 3 or not isinstance(args[0], DatetimeArray):
        raise MatlabError("dateshift expects dateshift(t, 'start' | 'end', unit) or dateshift(t, 'dayofweek', day).")
    t = args[0]
    mode = text(args[1]).lower()
    rule: Any = "current"
    if len(args) > 3:
        rule = text(args[3]).lower() if is_text(args[3]) else to_numeric(args[3]).astype(float)
    if mode == "dayofweek":
        target = args[2]
        if is_text(target):
            key = text(target).lower()
            if key == "weekday" or key == "weekend":
                raise MatlabError("dateshift: 'weekday' and 'weekend' targets are not supported; name a day.")
            matches = [k for k, w in enumerate(WEEKDAYS) if w.lower() == key or w[:3].lower() == key]
            if not matches:
                raise MatlabError(f"dateshift: unknown day name '{text(target)}'.")
            want = float(matches[0])
        else:
            want = float(to_numeric(target).flat[0]) - 1
        day0 = np.floor(t.data / DAY)
        tod = t.data - day0 * DAY
        dow = np.mod(day0 + 4, 7)
        ahead = np.mod(want - dow, 7)
        if isinstance(rule, str):
            if rule in ("next", "current"):
                delta = ahead
            elif rule == "previous":
                delta = ahead - np.where(ahead == 0, 0, 7)
            elif rule == "nearest":
                delta = np.where(ahead <= 3, ahead, ahead - 7)
            else:
                raise MatlabError(f"dateshift: unknown rule '{rule}'.")
        else:
            delta = ahead + 7 * (rule - 1)
        return t.like((day0 + delta) * DAY + tod)
    unit = text(args[2]).lower().rstrip("s") if text(args[2]).lower() != "s" else "second"
    if unit not in ("year", "quarter", "month", "week", "day", "hour", "minute", "second"):
        raise MatlabError(f"dateshift: unknown unit '{text(args[2])}'.")
    if isinstance(rule, str):
        steps = {"current": 0.0, "next": 1.0, "previous": -1.0, "nearest": None}.get(rule, "bad")
        if steps == "bad":
            raise MatlabError(f"dateshift: unknown rule '{rule}'.")
    else:
        steps = rule
    if mode not in ("start", "end"):
        raise MatlabError("dateshift: the second input must be 'start', 'end' or 'dayofweek'.")

    def shifted(k):
        start = _unit_start(t.data, unit, k)
        if mode == "start":
            return start
        nxt = _unit_start(t.data, unit, k + 1)
        # the end of a day-or-shorter unit is the instant the next one begins;
        # for longer units it is the start of the unit's last day
        return nxt if unit in ("day", "hour", "minute", "second") else nxt - DAY

    if steps is None:  # nearest
        lo = _unit_start(t.data, unit, 0.0)
        hi = _unit_start(t.data, unit, 1.0)
        with np.errstate(invalid="ignore"):
            near_hi = (t.data - lo) >= (hi - t.data)
        if mode == "start":
            return t.like(np.where(near_hi, hi, lo))
        return t.like(np.where(near_hi, shifted(0.0), shifted(-1.0)))
    return t.like(shifted(steps))


builtin("dateshift")(_dateshift)


@native("between", "caldiff")
def _between(interp, args, nargout):
    if len(args) == 1:  # caldiff(t)
        t = args[0]
        a, b = t.data.flatten(order="F")[:-1], t.data.flatten(order="F")[1:]
    else:
        a, b = _bc(_as_seconds(args[0])[0], _as_seconds(args[1])[0], "-")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    months = np.zeros(a.shape)
    for idx in np.ndindex(a.shape):
        x, y = a[idx], b[idx]
        if x != x or y != y:
            months[idx] = np.nan
            continue
        fx, fy = fields_from_seconds(x), fields_from_seconds(y)
        k = (float(fy[0]) - float(fx[0])) * 12 + float(fy[1]) - float(fx[1])
        sign = 1 if y >= x else -1
        while sign * (float(add_months(x, k)) - y) > 0:
            k -= sign
        months[idx] = k
    mid = add_months(a, months)
    with np.errstate(invalid="ignore"):
        days = np.trunc((b - mid) / DAY)
    return CalendarDuration(as_2d(months), as_2d(days), as_2d(b - mid - days * DAY))


builtin("between")(_between)
builtin("caldiff")(_between)


def _numeric_call(name: str):
    def f(interp, args, nargout):
        conv = []
        for a in args:
            if isinstance(a, DatetimeArray):
                conv.append(a.data / DAY + EPOCH_DATENUM)  # serial date numbers
            elif isinstance(a, DurationArray):
                conv.append(a.data / _UNIT_SECONDS.get(a.format, 1.0))
            else:
                conv.append(a)
        res = interp.call_named(name, conv, nargout, interp.frame)
        return tuple(res) if len(res) != 1 else res[0]
    return f


# Plotting and a few numeric helpers take datetimes as serial date numbers and durations as plain numbers.
for _n in ("plot", "scatter", "stairs", "stem", "bar", "barh", "area", "line", "semilogy", "semilogx", "loglog",
           "plot3", "fill", "patch", "errorbar", "xlim", "ylim", "xline", "yline", "histogram", "histcounts",
           "interp1", "polyfit", "trapz", "cumtrapz", "xticks", "yticks"):
    _NATIVE[_n] = _numeric_call(_n)


# -- serial date numbers and date text ------------------------------------------------------------------

def _datenum_values(args: list) -> np.ndarray:
    """Serial date numbers from any of the accepted input forms."""
    first = args[0]
    if isinstance(first, DatetimeArray):
        return first.data / DAY + EPOCH_DATENUM
    if isinstance(first, DurationArray):
        return first.data / DAY
    items = _text_items(first)
    if items is not None:
        fmt = _classic_to_modern(text(args[1])) if len(args) > 1 and is_text(args[1]) else None
        vals, _ = parse_texts(items[0], fmt)
        return vals.reshape(items[1], order="F") / DAY + EPOCH_DATENUM
    nums = [_numeric_arg(a, "Date values") for a in args]
    if len(nums) == 1:
        v = nums[0]
        if v.ndim == 2 and v.shape[1] in (3, 6) and v.size and _looks_like_datevec(v):
            parts = [v[:, k:k + 1] for k in range(v.shape[1])]
            return seconds_from_fields(*parts) / DAY + EPOCH_DATENUM
        return v
    if len(nums) > 6:
        raise MatlabError("datenum: too many inputs (use Y, M, D, H, MI, S).")
    parts = _broadcast_parts(nums, "datenum")
    while len(parts) < 3:
        parts.append(np.ones(parts[0].shape))
    return seconds_from_fields(*parts) / DAY + EPOCH_DATENUM


def _looks_like_datevec(v: np.ndarray) -> bool:
    if not np.all(np.isfinite(v)):
        return False
    return bool(np.all((v[:, 1] >= 1) & (v[:, 1] <= 12)) and np.all((v[:, 2] >= 1) & (v[:, 2] <= 31))
                and np.all(v[:, 0] == np.round(v[:, 0])) and np.all(np.abs(v[:, 0]) < 10000)
                and (v.shape[1] == 3 or (np.all((v[:, 3] >= 0) & (v[:, 3] < 25))
                                         and np.all((v[:, 4] >= 0) & (v[:, 4] < 61)))))


@native("datenum")
def _datenum(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for datenum.")
    return _datenum_values(args)


builtin("datenum")(_datenum)

_CLASSIC_FORMATS = {
    0: "dd-mmm-yyyy HH:MM:SS", 1: "dd-mmm-yyyy", 2: "mm/dd/yy", 3: "mmm", 4: "m", 5: "mm", 6: "mm/dd", 7: "dd",
    8: "ddd", 9: "d", 10: "yyyy", 11: "yy", 12: "mmmyy", 13: "HH:MM:SS", 14: "HH:MM:SS PM", 15: "HH:MM",
    16: "HH:MM PM", 17: "QQ-YY", 18: "QQ", 19: "dd/mm", 20: "dd/mm/yy", 21: "mmm.dd,yyyy HH:MM:SS",
    22: "mmm.dd,yyyy", 23: "mm/dd/yyyy", 24: "dd/mm/yyyy", 25: "yy/mm/dd", 26: "yyyy/mm/dd", 27: "QQ-YYYY",
    28: "mmmyyyy", 29: "yyyy-mm-dd", 30: "yyyymmddTHHMMSS", 31: "yyyy-mm-dd HH:MM:SS",
}

_CLASSIC_TOKEN = re.compile(r"yyyy|YYYY|yy|YY|mmmm|mmm|mm|m|dddd|ddd|dd|d|HH|MM|SS|FFF|AM|PM|QQ", )


def format_classic(secs: float, fmt: str) -> str:
    """Format with datestr-style codes (mm = month, MM = minute, PM switches to a 12-hour clock)."""
    if secs != secs:
        return "NaN"
    secs = math.floor(secs * 1000 + 0.5) / 1000
    y, mo, d, h, mi, s = (float(x) for x in fields_from_seconds(secs))
    twelve = "PM" in fmt or "AM" in fmt
    w = int(weekday_index(secs))

    def rep(m: re.Match) -> str:
        t = m.group(0)
        if t in ("yyyy", "YYYY"):
            return f"{int(y):04d}"
        if t in ("yy", "YY"):
            return f"{int(y) % 100:02d}"
        if t == "mmmm":
            return MONTHS[int(mo) - 1]
        if t == "mmm":
            return MONTHS[int(mo) - 1][:3]
        if t == "mm":
            return f"{int(mo):02d}"
        if t == "m":
            return MONTHS[int(mo) - 1][0]
        if t == "dddd":
            return WEEKDAYS[w]
        if t == "ddd":
            return WEEKDAYS[w][:3]
        if t == "dd":
            return f"{int(d):02d}"
        if t == "d":
            return WEEKDAYS[w][0]
        if t == "HH":
            return f"{(int(h) + 11) % 12 + 1:2d}" if twelve else f"{int(h):02d}"
        if t == "MM":
            return f"{int(mi):02d}"
        if t == "SS":
            return f"{int(math.floor(s + 1e-9)):02d}"
        if t == "FFF":
            return f"{int(round((s - math.floor(s + 1e-9)) * 1000)):03d}"[:3]
        if t in ("AM", "PM"):
            return "AM" if h < 12 else "PM"
        return f"Q{(int(mo) - 1) // 3 + 1}"
    return _CLASSIC_TOKEN.sub(rep, fmt)


def _classic_to_modern(fmt: str) -> str:
    """Translate a datestr-style format to datetime letters, for parsing text."""
    table = {"yyyy": "yyyy", "YYYY": "yyyy", "yy": "yy", "YY": "yy", "mmmm": "MMMM", "mmm": "MMM", "mm": "MM",
             "dddd": "eeee", "ddd": "eee", "dd": "dd", "HH": "HH", "MM": "mm", "SS": "ss", "FFF": "SSS", "AM": "a",
             "PM": "a"}
    twelve = "PM" in fmt or "AM" in fmt
    out, pos = [], 0
    for m in _CLASSIC_TOKEN.finditer(fmt):
        lit = fmt[pos:m.start()]
        if lit:
            out.append("".join(f"'{c}'" if c.isalpha() else c for c in lit))
        t = m.group(0)
        out.append("hh" if (t == "HH" and twelve) else table.get(t, f"'{t}'"))
        pos = m.end()
    lit = fmt[pos:]
    if lit:
        out.append("".join(f"'{c}'" if c.isalpha() else c for c in lit))
    return "".join(out)


@native("datestr")
def _datestr(interp, args, nargout):
    if not args:
        secs = np.array([now_seconds()])
        rest: list = []
    else:
        first = args[0]
        rest = list(args[1:])
        if isinstance(first, DatetimeArray):
            secs = first.data.flatten(order="F")
        elif _text_items(first) is not None:
            secs = ((_datenum_values([first]) - EPOCH_DATENUM) * DAY).flatten(order="F")
        else:
            v = _numeric_arg(first, "The date input")
            if v.ndim == 2 and v.shape[1] in (3, 6) and v.size and _looks_like_datevec(v):
                secs = seconds_from_fields(*[v[:, k] for k in range(v.shape[1])])
            else:
                secs = (v.flatten(order="F") - EPOCH_DATENUM) * DAY
            secs = np.round(secs * 1000) / 1000
    fmt: str | None = None
    if rest:
        f = rest[0]
        if is_text(f):
            fmt = text(f)
        else:
            code = int(to_numeric(f).flat[0])
            if code == -1:
                fmt = None
            elif code not in _CLASSIC_FORMATS:
                raise MatlabError(f"datestr: unknown format number {code} (use 0 to 31).")
            else:
                fmt = _CLASSIC_FORMATS[code]
    if fmt is None:
        finite = secs[np.isfinite(secs)]
        fmt = _CLASSIC_FORMATS[1] if finite.size and np.all(np.mod(finite, DAY) == 0) else _CLASSIC_FORMATS[0]
    return _char_rows([format_classic(float(s), fmt) for s in secs])


builtin("datestr")(_datestr)


@native("datevec")
def _datevec(interp, args, nargout):
    first = args[0]
    if isinstance(first, DatetimeArray):
        secs = first.data.flatten(order="F")
    elif isinstance(first, DurationArray):
        secs = first.data.flatten(order="F") - EPOCH_DATENUM * DAY
    elif _text_items(first) is not None:
        items = _text_items(first)
        fmt = _classic_to_modern(text(args[1])) if len(args) > 1 and is_text(args[1]) else None
        secs = parse_texts(items[0], fmt)[0]
    else:
        # serial date numbers resolve about 1e-5 s: snap to 0.1 ms so 10:30:45 does not come back as 44.99999
        secs = np.round((_datenum_values(list(args)).flatten(order="F") - EPOCH_DATENUM) * DAY * 1e4) / 1e4
    parts = fields_from_seconds(secs)
    if nargout > 1:
        return tuple(p.reshape(-1, 1) for p in parts[:nargout])
    return np.column_stack(parts) if secs.size else np.zeros((0, 6))


builtin("datevec")(_datevec)


@builtin("now")
def _now(interp, args, nargout):
    return now_seconds() / DAY + EPOCH_DATENUM


@builtin("date")
def _date(interp, args, nargout):
    return char(format_classic(math.floor(now_seconds() / DAY) * DAY, _CLASSIC_FORMATS[1]))


@builtin("today")
def _today(interp, args, nargout):
    return math.floor(now_seconds() / DAY) + EPOCH_DATENUM


@builtin("clock")
def _clock(interp, args, nargout):
    parts = fields_from_seconds(now_seconds())
    return np.array([[float(p) for p in parts]])


@builtin("eomday")
def _eomday(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("eomday expects eomday(year, month).")
    y, m = _bc(_numeric_arg(args[0], "The year"), _numeric_arg(args[1], "The month"), "eomday")
    if np.any((m < 1) | (m > 12) | (m != np.round(m))):
        raise MatlabError("eomday: months must be whole numbers from 1 to 12.")
    return as_2d(np.asarray(month_length(y, m)))


@builtin("addtodate")
def _addtodate(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("addtodate expects addtodate(dateNumber, quantity, unit).")
    d = _numeric_arg(args[0], "The date number")
    q = _numeric_arg(args[1], "The quantity")
    unit = text(args[2]).lower()
    secs = (d - EPOCH_DATENUM) * DAY
    if unit in ("year", "month"):
        if np.any(q != np.round(q)):
            raise MatlabError("addtodate: years and months must be whole numbers.")
        a, b = _bc(secs, q * (12 if unit == "year" else 1), "addtodate")
        return as_2d(np.asarray(add_months(a, b))) / DAY + EPOCH_DATENUM
    sizes = {"week": 7.0, "day": 1.0, "hour": 1 / 24, "minute": 1 / 1440, "second": 1 / 86400,
             "millisecond": 1 / 86400000}
    if unit not in sizes:
        raise MatlabError(f"addtodate: unknown unit '{text(args[2])}'. Use 'year', 'month', 'day', 'hour', "
                          "'minute', 'second' or 'millisecond'.")
    return ops.binop("+", d, q * sizes[unit])


@builtin("calendar")
def _calendar(interp, args, nargout):
    if not args:
        y, m = (float(x) for x in fields_from_seconds(now_seconds())[:2])
    elif len(args) == 1:
        y, m = (float(np.asarray(x).flat[0]) for x in fields_from_seconds(_as_seconds(args[0])[0])[:2])
    else:
        y, m = float(to_numeric(args[0]).flat[0]), float(to_numeric(args[1]).flat[0])
    first = int(np.mod(days_from_civil(y, m, 1) + 4, 7))
    n = int(month_length(y, m))
    grid = np.zeros(42)
    grid[first:first + n] = np.arange(1, n + 1)
    grid = grid.reshape(6, 7)
    if nargout:
        return grid
    title = f"{MONTHS[int(m) - 1][:3]} {int(y)}"
    lines = [title.center(34).rstrip(), "     S     M    Tu     W    Th     F     S"]
    lines += ["".join(f"{int(v):6d}" for v in row) for row in grid]
    interp.out("\n".join(lines) + "\n")
    return None
