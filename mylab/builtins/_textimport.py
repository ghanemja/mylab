"""Importing delimited text: the engine behind readtable, readmatrix, readcell, detectImportOptions
and the text datastores, plus the field scanner used by textscan.

The reading steps are the usual ones for delimited files: split the text into lines, find the
delimiter (the candidate that gives the most lines the same number of fields), find where the
data starts and whether a line of names sits above it, then give every column one type (number,
datetime, duration or text) and convert it.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, StringArray, Struct, char, is_char, string_result, to_int, to_str
from ._util import is_text, text

# -- small helpers ---------------------------------------------------------------------------------

_DELIM_WORDS = {"comma": ",", "space": " ", "tab": "\t", "semi": ";", "bar": "|", "\\t": "\t", ",": ",",
                ";": ";", "|": "|", " ": " ", "\t": "\t"}


def delimiter_list(v: Any) -> list[str]:
    """Delimiters from a char vector, string or cell: names such as 'tab' and escapes are understood."""
    if isinstance(v, Cell):
        items = [to_str(x, "delimiter") for x in v.values_f()]
    elif isinstance(v, StringArray):
        items = ["" if x is None else x for x in v.data.flatten(order="F")]
    else:
        items = [text(v, "delimiter")]
    out = []
    for d in items:
        d = _DELIM_WORDS.get(d.lower(), d).replace("\\t", "\t").replace("\\n", "\n").replace("\\r", "\r")
        if d == "":
            raise MatlabError("A delimiter must have at least one character.")
        out.append(d)
    return out


def text_items(v: Any, what: str) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x, what) for x in v.values_f()]
    if isinstance(v, StringArray):
        return ["" if x is None else x for x in v.data.flatten(order="F")]
    if isinstance(v, MString) or (is_char(v) and v.shape[0] <= 1):
        return [to_str(v)]
    raise MatlabError(f"{what} must be text or a cell array of text.")


def read_file_text(interp, fname: str, encoding: str | None = None) -> str:
    from .fileformats import find_file
    path = find_file(interp, fname)
    if path is None:
        raise MatlabError(f"Unable to find or open '{fname}'. Check the path and filename or file permissions.",
                          "MATLAB:textio:textio:FileNotFound")
    enc = encoding or "utf-8-sig"
    if enc.lower() in ("system", "utf-8", "utf8"):
        enc = "utf-8-sig"
    try:
        with open(path, encoding=enc, newline="") as fh:
            return fh.read()
    except LookupError:
        raise MatlabError(f"Unsupported encoding '{encoding}'.") from None
    except UnicodeDecodeError:
        with open(path, encoding="latin-1", newline="") as fh:
            return fh.read()
    except OSError as e:
        raise MatlabError(f"Unable to read '{fname}': {e.strerror or e}.") from None


def split_lines(content: str, endings: list[str] | None = None) -> list[str]:
    if endings:
        pattern = "|".join(re.escape(e) for e in sorted(endings, key=len, reverse=True))
        lines = re.split(pattern, content)
    else:
        lines = re.split(r"\r\n|\n|\r", content)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def split_fields(line: str, delims: list[str] | None, join: bool = False) -> list[str]:
    """Fields of one line. ``delims`` None means runs of white space. Double quotes group a field."""
    if delims is None or delims == [" "]:
        out = []
        for m in re.finditer(r'"((?:[^"]|"")*)"|\S+', line):
            out.append(m.group(1).replace('""', '"') if m.group(1) is not None else m.group(0))
        return out
    out = []
    cur: list[str] = []
    i, n = 0, len(line)
    first = [d[0] for d in delims]
    while i < n:
        ch = line[i]
        if ch == '"' and not "".join(cur).strip():
            j = i + 1
            buf = []
            while j < n:
                if line[j] == '"':
                    if j + 1 < n and line[j + 1] == '"':
                        buf.append('"')
                        j += 2
                        continue
                    break
                buf.append(line[j])
                j += 1
            cur = ["".join(buf)]
            i = j + 1
            continue
        hit = None
        if ch in first:
            hit = next((d for d in delims if line.startswith(d, i)), None)
        if hit is not None:
            out.append("".join(cur))
            cur = []
            i += len(hit)
            if join:
                while True:
                    more = next((d for d in delims if line.startswith(d, i)), None)
                    if more is None:
                        break
                    i += len(more)
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [f.strip(" \t") if f[:1] in " \t" or f[-1:] in " \t" else f for f in out]


def sniff_delimiter(lines: list[str]) -> list[str] | None:
    """The delimiter that splits the most lines into the same number (more than one) of fields."""
    sample = [ln for ln in lines if ln.strip()][:200]
    if not sample:
        return [","]
    best, best_score = None, 0.0
    for d in (",", "\t", ";", "|", " "):
        counts = [len(split_fields(ln, [d] if d != " " else None)) for ln in sample]
        tail = counts[-50:]
        mode = max(set(tail), key=tail.count)
        if mode < 2:
            continue
        score = counts.count(mode) / len(counts) + (0.001 if d != " " else 0.0)
        if score > best_score:
            best, best_score = d, score
    if best is None:
        return [","]
    return None if best == " " else [best]


_NUM = r"[+-]?(?:\d+\.?\d*(?:[eEdD][+-]?\d+)?|\.\d+(?:[eEdD][+-]?\d+)?|inf(?:inity)?|nan)"
_NUM_RE = re.compile(rf"^{_NUM}$", re.IGNORECASE)
_COMPLEX_RE = re.compile(rf"^({_NUM})?\s*([+-]\s*(?:\d+\.?\d*(?:[eEdD][+-]?\d+)?|\.\d+(?:[eEdD][+-]?\d+)?)?)[ij]$",
                         re.IGNORECASE)
_HEX_RE = re.compile(r"^0[xX][0-9a-fA-F]+$")
_DURATION_RE = re.compile(r"^-?(\d+:)?\d+:\d\d(:\d\d(\.\d+)?)?$")


def parse_number(s: str, decimal: str = ".", thousands: str = "", trim: bool = False) -> Any:
    """float, complex or None (not a number). Empty text is NaN."""
    t = s.strip()
    if t == "":
        return float("nan")
    if thousands:
        t = t.replace(thousands, "")
    if decimal != ".":
        if "." in t:
            return None
        t = t.replace(decimal, ".")
    if trim:
        m = re.search(_NUM, t, re.IGNORECASE)
        if m is None:
            return None
        t = m.group(0)
    if _NUM_RE.match(t):
        return float(t.replace("d", "e").replace("D", "e").replace("infinity", "inf").replace("Infinity", "inf"))
    m = _COMPLEX_RE.match(t)
    if m:
        re_part = float(m.group(1).replace("d", "e").replace("D", "e")) if m.group(1) else 0.0
        im_text = m.group(2).replace(" ", "").replace("d", "e").replace("D", "e")
        im_part = float(im_text + "1") if im_text in ("+", "-") else float(im_text)
        return complex(re_part, im_part)
    return None


def try_datetimes(items: list[str]) -> tuple[np.ndarray, bool] | None:
    """Seconds since 1970 when every non-empty item is a date in one recognized format, else None."""
    live = [s.strip() for s in items if s.strip()]
    if not live:
        return None
    if not all(re.search(r"\d", s) and re.search(r"[-/.: ,T]", s) for s in live[:50]):
        return None
    if all(re.fullmatch(r"[\d.:]+", s) for s in live[:50]):
        return None  # plain times and dotted numbers are not dates
    from .datetimes import parse_texts
    try:
        secs, has_time = parse_texts([s.strip() for s in items], None)
    except MatlabError:
        return None
    blank = np.array([s.strip() == "" for s in items])
    if np.any(np.isnan(secs) & ~blank):
        return None
    return secs, has_time


def parse_durations(items: list[str]) -> np.ndarray | None:
    from .datetimes import _parse_duration_text
    live = [s.strip() for s in items if s.strip()]
    if not live or not all(_DURATION_RE.match(s) for s in live):
        return None
    return np.array([_parse_duration_text(s) if s.strip() else np.nan for s in items], dtype=float)


# -- name-value options ----------------------------------------------------------------------------

def name_values(args: list, known: dict, fn: str) -> dict:
    """{canonical name: value}. ``known`` maps lower-case names to canonical names (None = unsupported)."""
    if len(args) % 2:
        raise MatlabError(f"{fn}: options must be name-value pairs.")
    out = {}
    for k in range(0, len(args), 2):
        if not is_text(args[k]):
            raise MatlabError(f"{fn}: option names must be text.")
        key = text(args[k]).lower()
        if key not in known:
            close = [n for n in known.values() if n and n.lower().startswith(key)]
            if len(close) == 1 and len(key) >= 3:
                out[close[0]] = args[k + 1]
                continue
            raise MatlabError(f"{fn}: unknown option '{text(args[k])}'.")
        if known[key] is None:
            raise MatlabError(f"{fn}: the option '{text(args[k])}' is not supported.")
        out[known[key]] = args[k + 1]
    return out


def _known(*names: str, unsupported: tuple = ()) -> dict:
    d = {n.lower(): n for n in names}
    d.update({n.lower(): None for n in unsupported})
    return d


def _flag(v: Any) -> bool:
    if is_text(v):
        t = text(v).lower()
        if t in ("on", "true"):
            return True
        if t in ("off", "false"):
            return False
        raise MatlabError(f"Expected true or false, not '{text(v)}'.")
    return bool(np.asarray(v).any())


# -- import options --------------------------------------------------------------------------------

_TYPES = ("double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
          "logical", "char", "string", "categorical", "datetime", "duration")
_RULES = {"MissingRule": ("fill", "error", "omitrow", "omitvar"),
          "ImportErrorRule": ("fill", "error", "omitrow", "omitvar"),
          "EmptyLineRule": ("skip", "read"),
          "ConsecutiveDelimitersRule": ("split", "join", "error"),
          "LeadingDelimitersRule": ("keep", "ignore", "error"),
          "TrailingDelimitersRule": ("keep", "ignore", "error"),
          "ExtraColumnsRule": ("addvars", "ignore", "wrap", "error"),
          "VariableNamingRule": ("modify", "preserve")}
_VAR_OPTS = ("FillValue", "TreatAsMissing", "DecimalSeparator", "ThousandsSeparator", "TrimNonNumeric",
             "InputFormat", "DatetimeFormat", "TimeZone", "Categories", "Ordinal", "TrueSymbols", "FalseSymbols",
             "WhitespaceRule", "EmptyFieldRule", "QuoteRule", "Prefixes", "Suffixes", "Type", "Name",
             "DurationFormat", "ExponentCharacter", "NumberSystem", "DatetimeLocale", "FieldSeparator",
             "Protected")


class ImportOptions(MObject):
    """What detectImportOptions returns: how a delimited text file is to be read."""

    _METHODS = {"setvartype", "setvaropts", "getvaropts", "preview"}

    def __init__(self):
        self.file = ""
        self.delimiter: list[str] | None = [","]
        self.names: list[str] = []
        self.types: list[str] = []
        self.selected: list[str] | None = None  # None = all
        self.var_opts: list[dict] = []
        self.props: dict[str, Any] = {
            "DataLines": np.array([[1.0, np.inf]]), "VariableNamesLine": 0.0, "RowNamesColumn": 0.0,
            "VariableUnitsLine": 0.0, "VariableDescriptionsLine": 0.0, "Whitespace": "\b\t ",
            "LineEnding": ["\n", "\r", "\r\n"], "CommentStyle": [], "Encoding": "UTF-8",
            "MissingRule": "fill", "ImportErrorRule": "fill", "EmptyLineRule": "skip",
            "ConsecutiveDelimitersRule": "split", "LeadingDelimitersRule": "keep",
            "TrailingDelimitersRule": "ignore", "ExtraColumnsRule": "addvars", "VariableNamingRule": "modify",
        }
        self.text_type = "char"
        self.datetime_type = "datetime"
        self.range: Any = None  # the Range the options were detected with, if any

    def class_name(self) -> str:
        return "matlab.io.text.DelimitedTextImportOptions"

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "matlab.io.ImportOptions")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def copy(self) -> "ImportOptions":
        o = ImportOptions()
        o.file = self.file
        o.delimiter = None if self.delimiter is None else list(self.delimiter)
        o.names, o.types = list(self.names), list(self.types)
        o.selected = None if self.selected is None else list(self.selected)
        o.var_opts = [dict(v) for v in self.var_opts]
        o.props = dict(self.props)
        o.text_type, o.datetime_type = self.text_type, self.datetime_type
        o.range = self.range
        return o

    # variables ----------------------------------------------------------------------------------
    def positions(self, sel: Any, what: str = "variable") -> list[int]:
        if isinstance(sel, np.ndarray) and sel.dtype == bool:
            flags = sel.flatten(order="F")
            if flags.size != len(self.names):
                raise MatlabError("A logical selection must have one element per variable.")
            return [k for k in range(len(self.names)) if flags[k]]
        if isinstance(sel, np.ndarray) and sel.dtype.kind in "iuf":
            idx = [int(x) - 1 for x in sel.flatten(order="F")]
            if any(k < 0 or k >= len(self.names) for k in idx):
                raise MatlabError(f"A {what} index is out of range; the file has {len(self.names)} variables.")
            return idx
        out = []
        for n in text_items(sel, f"The {what} selection"):
            if n not in self.names:
                raise MatlabError(f"Unknown variable name: '{n}'.", "MATLAB:textio:io:UnknownVarName")
            out.append(self.names.index(n))
        return out

    def selected_positions(self) -> list[int]:
        if self.selected is None:
            return list(range(len(self.names)))
        return [self.names.index(n) for n in self.selected if n in self.names]

    # properties ---------------------------------------------------------------------------------
    def get_field(self, interp, name: str) -> Any:
        if name == "Delimiter":
            return Cell.from_list([char(d) for d in (self.delimiter or [" "])])
        if name == "VariableNames":
            return Cell.from_list([char(n) for n in self.names])
        if name == "VariableTypes":
            return Cell.from_list([char(t) for t in self.types])
        if name == "SelectedVariableNames":
            return Cell.from_list([char(self.names[k]) for k in self.selected_positions()])
        if name == "VariableOptions":
            return self._var_structs(list(range(len(self.names))))
        if name == "PreserveVariableNames":
            return np.array([[self.props["VariableNamingRule"] == "preserve"]])
        if name in self.props:
            v = self.props[name]
            if isinstance(v, list):
                return Cell.from_list([char(x) for x in v])
            if isinstance(v, str):
                return char(v)
            return np.array([[float(v)]]) if isinstance(v, (int, float)) else v
        if name == "TextType":
            return char(self.text_type)
        if name == "DatetimeType":
            return char(self.datetime_type)
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def set_field(self, interp, name: str, value: Any) -> "ImportOptions":
        o = self.copy()
        if name == "Delimiter":
            o.delimiter = delimiter_list(value)
            if o.delimiter == [" "]:
                o.delimiter = None
        elif name == "VariableNames":
            names = text_items(value, "VariableNames")
            if len(names) != len(self.names):
                raise MatlabError(f"VariableNames must have {len(self.names)} names, one per variable.")
            if len(set(names)) != len(names):
                raise MatlabError("VariableNames must be unique.")
            if o.selected is not None:
                o.selected = [names[self.names.index(n)] for n in o.selected]
            o.names = names
            for v, n in zip(o.var_opts, names):
                v["Name"] = n
        elif name == "VariableTypes":
            types = text_items(value, "VariableTypes")
            if len(types) != len(self.names):
                raise MatlabError(f"VariableTypes must have {len(self.names)} entries, one per variable.")
            for t in types:
                _check_type(t)
            o.types = types
            for v, t in zip(o.var_opts, types):
                v["Type"] = t
        elif name == "SelectedVariableNames":
            o.selected = [self.names[k] for k in self.positions(value)]
        elif name == "DataLines":
            a = np.asarray(value, dtype=float)
            if a.size == 1:
                a = np.array([[float(a.flat[0]), np.inf]])
            if a.ndim != 2 or a.shape[1] != 2 or np.any(a[:, 0] < 1):
                raise MatlabError("DataLines must be a first line number or rows of [first last].")
            o.props[name] = a
        elif name in ("VariableNamesLine", "RowNamesColumn", "VariableUnitsLine", "VariableDescriptionsLine"):
            n = to_int(value, name)
            if n < 0:
                raise MatlabError(f"{name} must not be negative.")
            o.props[name] = float(n)
        elif name in _RULES:
            v = text(value, name).lower()
            if v not in _RULES[name]:
                raise MatlabError(f"{name} must be one of: {', '.join(_RULES[name])}.")
            o.props[name] = v
        elif name == "PreserveVariableNames":
            o.props["VariableNamingRule"] = "preserve" if _flag(value) else "modify"
        elif name in ("LineEnding", "CommentStyle"):
            o.props[name] = [x.replace("\\n", "\n").replace("\\r", "\r") for x in text_items(value, name)]
        elif name in ("Whitespace", "Encoding"):
            o.props[name] = text(value, name)
        else:
            raise MatlabError(f"Unrecognized or read-only property '{name}' for class '{self.class_name()}'.")
        return o

    def _var_structs(self, idx: list[int]) -> Struct:
        fields = ["Name", "Type", "FillValue", "TreatAsMissing", "QuoteRule", "Prefixes", "Suffixes",
                  "EmptyFieldRule"]
        data = np.empty((1, len(idx)), dtype=object)
        for j, k in enumerate(idx):
            v = self.var_opts[k]
            d = {"Name": char(self.names[k]), "Type": char(self.types[k]),
                 "FillValue": v.get("FillValue", _default_fill(self.types[k])),
                 "TreatAsMissing": Cell.from_list([char(x) for x in v.get("TreatAsMissing", [])]),
                 "QuoteRule": char(v.get("QuoteRule", "remove")),
                 "Prefixes": Cell.from_list([char(x) for x in v.get("Prefixes", [])]),
                 "Suffixes": Cell.from_list([char(x) for x in v.get("Suffixes", [])]),
                 "EmptyFieldRule": char(v.get("EmptyFieldRule", "missing"))}
            for extra in ("DecimalSeparator", "ThousandsSeparator", "InputFormat", "DatetimeFormat", "TimeZone",
                          "DurationFormat"):
                if extra in v:
                    d[extra] = char(v[extra])
            data[0, j] = d
        all_fields = list(fields)
        for d in data.flat:
            for f in d:
                if f not in all_fields:
                    all_fields.append(f)
        for d in data.flat:
            for f in all_fields:
                d.setdefault(f, np.zeros((0, 0)))
        return Struct(data, all_fields)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name == "setvartype":
            return [set_var_type(self, rest)]
        if name == "setvaropts":
            return [set_var_opts(self, rest)]
        if name == "getvaropts":
            idx = self.positions(rest[0]) if rest else list(range(len(self.names)))
            return [self._var_structs(idx)]
        if name == "preview":
            if not rest or not is_text(rest[0]):
                raise MatlabError("Use preview(filename, opts) to see the first rows of a file.")
            from .table import TableObject, _rows
            t = read_table(interp, [rest[0], self], "preview")
            pos = np.arange(min(8, t.nrows))
            return [TableObject(list(t.names), [_rows(c, pos) for c in t.columns])]
        return super().call_method(interp, name, args, nargout)

    def display_lines(self) -> list[str]:
        def cell(items: list) -> str:
            return "{" + ", ".join(f"'{x}'" for x in items) + "}"
        d = [x.replace("\t", "\\t") for x in (self.delimiter or [" "])]
        first = float(self.props["DataLines"][0, 0])
        return ["  DelimitedTextImportOptions with properties:", "",
                "   Format Properties:",
                f"                    Delimiter: {cell(d)}",
                f"                   Whitespace: '{self.props['Whitespace'].encode('unicode_escape').decode()}'",
                f"                 CommentStyle: {cell(self.props['CommentStyle'])}",
                f"    ConsecutiveDelimitersRule: '{self.props['ConsecutiveDelimitersRule']}'",
                f"        LeadingDelimitersRule: '{self.props['LeadingDelimitersRule']}'",
                f"                EmptyLineRule: '{self.props['EmptyLineRule']}'",
                f"                     Encoding: '{self.props['Encoding']}'", "",
                "   Replacement Properties:",
                f"                  MissingRule: '{self.props['MissingRule']}'",
                f"              ImportErrorRule: '{self.props['ImportErrorRule']}'",
                f"             ExtraColumnsRule: '{self.props['ExtraColumnsRule']}'", "",
                "   Variable Import Properties: Set types by name using setvartype",
                f"                VariableNames: {cell(self.names)}",
                f"                VariableTypes: {cell(self.types)}",
                f"        SelectedVariableNames: {cell([self.names[k] for k in self.selected_positions()])}",
                "              VariableOptions: Show all with getvaropts",
                "	Access VariableOptions sub-properties using setvaropts/getvaropts",
                f"           VariableNamingRule: '{self.props['VariableNamingRule']}'", "",
                "   Location Properties:",
                f"                    DataLines: [{first:g} Inf]",
                f"            VariableNamesLine: {self.props['VariableNamesLine']:g}",
                f"               RowNamesColumn: {self.props['RowNamesColumn']:g}"]


def _check_type(t: str) -> str:
    if t not in _TYPES:
        raise MatlabError(f"'{t}' is not a variable type. Types are: {', '.join(_TYPES)}.")
    return t


def _default_fill(t: str) -> Any:
    if t in ("char",):
        return char("")
    if t in ("string", "categorical", "datetime", "duration"):
        return np.array([[np.nan]])
    if t == "logical":
        return np.array([[False]])
    return np.array([[np.nan]]) if t in ("double", "single") else np.array([[0.0]])


def set_var_type(opts: ImportOptions, rest: list) -> ImportOptions:
    if not rest or len(rest) > 2:
        raise MatlabError("setvartype(opts, type) or setvartype(opts, selection, type).")
    o = opts.copy()
    idx = o.positions(rest[0]) if len(rest) == 2 else list(range(len(o.names)))
    types = text_items(rest[-1], "The type")
    if len(types) == 1:
        types = types * len(idx)
    if len(types) != len(idx):
        raise MatlabError("setvartype: give one type, or one type per selected variable.")
    for k, t in zip(idx, types):
        o.types[k] = _check_type(t)
        o.var_opts[k]["Type"] = t
        o.var_opts[k].pop("FillValue", None)
    return o


def set_var_opts(opts: ImportOptions, rest: list) -> ImportOptions:
    o = opts.copy()
    if len(rest) % 2:
        idx = o.positions(rest[0])
        pairs = rest[1:]
    else:
        idx = list(range(len(o.names)))
        pairs = rest
    for k in range(0, len(pairs), 2):
        key = text(pairs[k], "option name")
        match = [n for n in _VAR_OPTS if n.lower() == key.lower()]
        if not match:
            raise MatlabError(f"setvaropts: unknown variable option '{key}'. Options are: {', '.join(_VAR_OPTS)}.")
        name = match[0]
        v = pairs[k + 1]
        if name in ("WhitespaceRule", "EmptyFieldRule", "QuoteRule", "ExponentCharacter", "NumberSystem",
                    "DatetimeLocale", "FieldSeparator", "Protected", "DurationFormat"):
            raise MatlabError(f"setvaropts: the variable option '{name}' is not supported.")
        if name == "Type":
            for j in idx:
                o.types[j] = _check_type(text(v))
                o.var_opts[j]["Type"] = o.types[j]
            continue
        if name == "Name":
            names = text_items(v, "Name")
            if len(names) != len(idx):
                raise MatlabError("setvaropts: give one name per selected variable.")
            new = list(o.names)
            for j, n in zip(idx, names):
                new[j] = n
            if len(set(new)) != len(new):
                raise MatlabError("Variable names must be unique.")
            if o.selected is not None:
                o.selected = [new[o.names.index(n)] for n in o.selected]
            o.names = new
            continue
        if name in ("TreatAsMissing", "Prefixes", "Suffixes", "Categories", "TrueSymbols", "FalseSymbols"):
            val: Any = text_items(v, name)
        elif name in ("DecimalSeparator", "ThousandsSeparator", "InputFormat", "DatetimeFormat", "TimeZone"):
            val = text(v, name)
        elif name in ("TrimNonNumeric", "Ordinal"):
            val = _flag(v)
        else:
            val = v
        for j in idx:
            if name in ("DecimalSeparator", "ThousandsSeparator", "TrimNonNumeric") and \
                    o.types[j] not in ("double", "single") and not o.types[j].startswith(("int", "uint")):
                raise MatlabError(f"setvaropts: '{name}' applies to numeric variables; '{o.names[j]}' is "
                                  f"{o.types[j]}.")
            if name in ("InputFormat", "DatetimeFormat", "TimeZone") and o.types[j] != "datetime":
                raise MatlabError(f"setvaropts: '{name}' applies to datetime variables; '{o.names[j]}' is "
                                  f"{o.types[j]}.")
            o.var_opts[j][name] = val
    return o


# -- reading rows ----------------------------------------------------------------------------------

def _strip_comments(lines: list[str], style: list[str]) -> list[str]:
    if not style:
        return lines
    if len(style) == 1:
        s = style[0]
        return [ln.split(s, 1)[0] if s in ln else ln for ln in lines]
    opener, closer = style[0], style[1]
    joined = "\n".join(lines)
    joined = re.sub(re.escape(opener) + r".*?" + re.escape(closer), lambda m: "\n" * m.group(0).count("\n"),
                    joined, flags=re.S)
    return joined.split("\n")


def _is_texty(fields: list[str], decimal: str, thousands: str) -> bool:
    live = [f for f in fields if f.strip()]
    return bool(live) and all(parse_number(f, decimal, thousands) is None for f in live)


class Layout:
    """The rows of a delimited file and where its names and data are."""

    def __init__(self, lines: list[str], delims: list[str] | None, join: bool, lead: str = "keep"):
        self.lines = lines
        self.delims = delims
        self.rows: list[list[str] | None] = []
        for ln in lines:
            if not ln.strip():
                self.rows.append(None)
                continue
            f = split_fields(ln, delims, join)
            if lead == "ignore" and delims is not None:
                while f and f[0] == "":
                    f = f[1:]
            self.rows.append(f)

    def width(self, start: int) -> int:
        counts = [len(r) for r in self.rows[start:] if r is not None]
        if not counts:
            return 0
        tail = counts[-200:]
        return max(set(tail), key=tail.count)


def detect_layout(rows: list, width: int, decimal: str, thousands: str, want_names: bool | None,
                  header_lines: int | None) -> tuple[int, int]:
    """(line with the variable names or -1, first data line), both counted from zero."""
    n = len(rows)
    first = 0
    if header_lines is not None:
        first = min(header_lines, n)
    else:
        # lines above the table: blank, or with another number of fields than the table has
        while first < n and (rows[first] is None or len(rows[first]) != width):
            first += 1
        if first >= n:
            first = 0  # no line has the usual number of fields: take everything as data
    while first < n and rows[first] is None:
        first += 1
    if first >= n:
        return -1, n
    names_line = -1
    row = rows[first]
    texty = _is_texty(row, decimal, thousands) and all(
        not try_datetimes([f]) and not _DURATION_RE.match(f.strip()) for f in row if f.strip())
    if want_names is True or (want_names is None and texty):
        names_line = first
        first += 1
    return names_line, first


_NA_WORDS = {"NA", "N/A", "na", "n/a", "NaN", "nan"}


def column_kind(items: list[str], decimal: str, thousands: str, missing: set, trim: bool = False) -> str:
    """'double', 'datetime', 'duration' or 'char' for the text of one column."""
    live = [s for s in items if s.strip() and s.strip() not in missing]
    if not live:
        return "double"
    # the usual "not available" marks do not turn a column of numbers into text; they are read as NaN
    numbers = [s for s in live if s.strip() not in _NA_WORDS]
    if numbers and all(parse_number(s, decimal, thousands, trim) is not None for s in numbers):
        return "double"
    if parse_durations(live) is not None:
        return "duration"
    if try_datetimes(live) is not None:
        return "datetime"
    return "char"


def _text_column(items: list, kind: str) -> Any:
    if kind == "string":
        data = np.empty((len(items), 1), dtype=object)
        for i, s in enumerate(items):
            data[i, 0] = s
        return StringArray(data)
    data = np.empty((len(items), 1), dtype=object)
    for i, s in enumerate(items):
        data[i, 0] = char(s or "")
    return Cell(data)


def convert_column(items: list[str], kind: str, vo: dict, defaults: dict, name: str) -> tuple[Any, np.ndarray]:
    """(column value, mask of rows whose field was missing or could not be converted)."""
    from .datetimes import DatetimeArray, DurationArray, parse_with_format
    n = len(items)
    missing_words = set(vo.get("TreatAsMissing", [])) | defaults["missing"]
    decimal = vo.get("DecimalSeparator", defaults["decimal"])
    thousands = vo.get("ThousandsSeparator", defaults["thousands"])
    trim = bool(vo.get("TrimNonNumeric", defaults["trim"]))
    pre, suf = vo.get("Prefixes", []), vo.get("Suffixes", [])
    cleaned = []
    for s in items:
        t = s.strip() if kind not in ("char", "string") else s
        for p in pre:
            if t.startswith(p):
                t = t[len(p):]
                break
        for p in suf:
            if p and t.endswith(p):
                t = t[:-len(p)]
                break
        cleaned.append(t)
    blank = np.array([t.strip() == "" or t.strip() in missing_words for t in cleaned], dtype=bool).reshape(n)
    bad = blank.copy()
    numeric = kind in ("double", "single") or kind.startswith(("int", "uint"))
    if numeric:
        vals = np.full(n, np.nan, dtype=complex)
        for i, t in enumerate(cleaned):
            if blank[i]:
                continue
            v = parse_number(t, decimal, thousands, trim)
            if v is None and _HEX_RE.match(t.strip()):
                v = float(int(t.strip(), 16))
            if v is None:
                bad[i] = True
            else:
                vals[i] = v
        fill = vo.get("FillValue")
        if fill is not None and bad.any():
            vals[bad] = complex(np.asarray(fill).flat[0])
        out = vals.real if not np.any(np.nan_to_num(vals.imag) != 0) else vals
        out = out.reshape(-1, 1)
        if kind == "single":
            out = out.astype(np.complex64 if out.dtype.kind == "c" else np.float32)
        elif kind != "double":
            from ..values import INT_TYPES, saturate_int
            out = saturate_int(np.real(out), INT_TYPES[kind])
        return out, bad
    if kind == "logical":
        true_words = {w.lower() for w in vo.get("TrueSymbols", ["true", "t", "1", "yes", "y"])}
        false_words = {w.lower() for w in vo.get("FalseSymbols", ["false", "f", "0", "no", "n"])}
        out = np.zeros((n, 1), dtype=bool)
        for i, t in enumerate(cleaned):
            if blank[i]:
                continue
            w = t.strip().lower()
            if w in true_words:
                out[i, 0] = True
            elif w not in false_words:
                v = parse_number(w)
                if v is None or v != v:
                    bad[i] = True
                else:
                    out[i, 0] = bool(v)
        return out, bad
    if kind == "datetime":
        fmt = vo.get("InputFormat")
        secs = np.full(n, np.nan)
        if fmt:
            for i, t in enumerate(cleaned):
                if not blank[i]:
                    secs[i] = parse_with_format(t.strip(), fmt)
            has_time = None
        else:
            live = [t if not blank[i] else "" for i, t in enumerate(cleaned)]
            res = try_datetimes(live)
            if res is not None:
                secs, has_time = res
            else:  # mixed formats: convert one by one
                from .datetimes import parse_texts
                for i, t in enumerate(live):
                    if t.strip():
                        try:
                            secs[i] = parse_texts([t], None)[0][0]
                        except MatlabError:
                            pass
        bad = bad | np.isnan(secs)
        return DatetimeArray(secs.reshape(-1, 1), vo.get("DatetimeFormat") or None, vo.get("TimeZone", "")), bad
    if kind == "duration":
        from .datetimes import _parse_duration_text
        secs = np.array([_parse_duration_text(t) if not blank[i] else np.nan for i, t in enumerate(cleaned)])
        bad = bad | np.isnan(secs)
        return DurationArray(secs.reshape(-1, 1), None), bad
    if kind == "categorical":
        from .categorical import CategoricalArray
        texts = [None if blank[i] else t.strip() for i, t in enumerate(cleaned)]
        cats = vo.get("Categories") or sorted({t for t in texts if t is not None})
        codes = np.array([np.nan if t is None or t not in cats else cats.index(t) + 1.0 for t in texts])
        return CategoricalArray(codes.reshape(-1, 1), list(cats), bool(vo.get("Ordinal", False))), bad
    # text
    fill = vo.get("FillValue")
    fill_text = to_str(fill) if fill is not None and (is_char(fill) or isinstance(fill, MString)) else None
    texts = []
    for i, t in enumerate(cleaned):
        if blank[i]:
            texts.append(fill_text if fill_text is not None else (None if kind == "string" else ""))
        else:
            texts.append(t)
    return _text_column(texts, kind), bad


def valid_names(raw: list[str], rule: str) -> list[str]:
    from .sysmore import make_valid_name
    out: list[str] = []
    for k, h in enumerate(raw):
        h = h.strip()
        if rule == "preserve":
            name = h or f"Var{k + 1}"
        else:
            name = make_valid_name(h, "underscore", "x") if h else f"Var{k + 1}"
        base, n = name, 1
        while name in out:
            name = f"{base}_{n}"
            n += 1
        out.append(name)
    return out


_COMMON = ("Delimiter", "NumHeaderLines", "HeaderLines", "ReadVariableNames", "VariableNamingRule",
           "PreserveVariableNames", "TextType", "DatetimeType", "DurationType", "DecimalSeparator",
           "ThousandsSeparator", "CommentStyle", "Encoding", "FileType", "ExpectedNumVariables", "Whitespace",
           "LineEnding", "TreatAsMissing", "TreatAsEmpty", "MissingRule", "ImportErrorRule", "EmptyLineRule",
           "ConsecutiveDelimitersRule", "LeadingDelimitersRule", "TrailingDelimitersRule", "ExtraColumnsRule",
           "TrimNonNumeric", "Range", "DataLines", "VariableNamesLine", "RowNamesColumn", "ReadRowNames",
           "MultipleDelimsAsOne", "Format", "EmptyValue", "VariableNames", "VariableTypes", "DateLocale",
           "VariableNamesRange", "RowNamesRange", "OutputType", "SelectedVariableNames", "VariableUnitsLine",
           "VariableDescriptionsLine", "EmptyColumnType")
_UNSUPPORTED = ("WebOptions", "Sheet", "UseExcel", "NumberSystem", "ExponentCharacter", "HexType", "BinaryType",
                "VariableWidths", "PartialFieldRule", "EmptyFieldRule", "TableSelector", "RowSelector",
                "VariableSelectors", "TableNodeName", "RowNodeName", "VariableNodeNames", "ImportAttributes",
                "AttributeSuffix", "RegisteredNamespaces", "MergeLines")
KNOWN = _known(*_COMMON, unsupported=_UNSUPPORTED)


def _range_bounds(v: Any, fn: str) -> tuple[int, int, int | None, int | None]:
    """(first row, first column, last row, last column), counted from zero; None = to the end."""
    if is_text(v):
        t = text(v).strip().upper()
        m = re.fullmatch(r"([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?", t)
        if m:
            col = lambda letters: sum((ord(c) - 64) * 26 ** k for k, c in enumerate(reversed(letters))) - 1
            r1, c1 = int(m.group(2)) - 1, col(m.group(1))
            if m.group(3):
                return r1, c1, int(m.group(4)) - 1, col(m.group(3))
            return r1, c1, None, None
        m = re.fullmatch(r"(\d+):(\d+)", t)
        if m:
            return int(m.group(1)) - 1, 0, int(m.group(2)) - 1, None
        m = re.fullmatch(r"([A-Z]+):([A-Z]+)", t)
        if m:
            col = lambda letters: sum((ord(c) - 64) * 26 ** k for k, c in enumerate(reversed(letters))) - 1
            return 0, col(m.group(1)), None, col(m.group(2))
        raise MatlabError(f"{fn}: the range '{text(v)}' is not understood. Use a corner such as 'B2', a block "
                          "such as 'A2:C5', rows '2:5', columns 'A:C' or [row1 col1 row2 col2].")
    r = [float(x) for x in np.asarray(v, dtype=float).flatten()]
    if len(r) == 2:
        return int(r[0]) - 1, int(r[1]) - 1, None, None
    if len(r) == 4:
        return (int(r[0]) - 1, int(r[1]) - 1, None if np.isinf(r[2]) else int(r[2]) - 1,
                None if np.isinf(r[3]) else int(r[3]) - 1)
    raise MatlabError(f"{fn}: a numeric range must be [row col] or [row1 col1 row2 col2].")


class Parsed:
    """A delimited file split into its name line and data rows, with settings shared by the readers."""

    def __init__(self, interp, fname: str, nv: dict, fn: str, names_default: bool | None):
        self.fn = fn
        enc = text(nv["Encoding"]) if "Encoding" in nv else None
        content = read_file_text(interp, fname, enc)
        endings = [x.replace("\\n", "\n").replace("\\r", "\r") for x in text_items(nv["LineEnding"], "LineEnding")] \
            if "LineEnding" in nv else None
        lines = split_lines(content, endings)
        style = text_items(nv["CommentStyle"], "CommentStyle") if "CommentStyle" in nv else []
        lines = _strip_comments(lines, style)
        self.comment_style = style
        if "FileType" in nv and text(nv["FileType"]).lower() not in ("text", "delimitedtext"):
            raise MatlabError(f"{fn}: FileType '{text(nv['FileType'])}' is not supported for this file; "
                              "use 'text' or 'delimitedtext'.")
        self.decimal = text(nv["DecimalSeparator"]) if "DecimalSeparator" in nv else "."
        self.thousands = text(nv["ThousandsSeparator"]) if "ThousandsSeparator" in nv else ""
        if self.decimal == self.thousands:
            raise MatlabError(f"{fn}: DecimalSeparator and ThousandsSeparator must differ.")
        if "Delimiter" in nv:
            self.delims = delimiter_list(nv["Delimiter"])
            if self.delims == [" "]:
                self.delims = None
        else:
            self.delims = sniff_delimiter(lines)
            if self.delims == [self.decimal]:
                self.delims = [";"] if any(";" in ln for ln in lines) else None
        join = False
        if "MultipleDelimsAsOne" in nv:
            join = _flag(nv["MultipleDelimsAsOne"])
        if "ConsecutiveDelimitersRule" in nv:
            rule = text(nv["ConsecutiveDelimitersRule"]).lower()
            if rule not in _RULES["ConsecutiveDelimitersRule"]:
                raise MatlabError(f"{fn}: ConsecutiveDelimitersRule must be 'split', 'join' or 'error'.")
            join = rule == "join"
            if rule == "error":
                for k, ln in enumerate(lines, 1):
                    f = split_fields(ln, self.delims)
                    if self.delims and any(x == "" for x in f[1:-1]):
                        raise MatlabError(f"{fn}: line {k} has consecutive delimiters.")
        lead = text(nv["LeadingDelimitersRule"]).lower() if "LeadingDelimitersRule" in nv else "keep"
        if lead not in _RULES["LeadingDelimitersRule"]:
            raise MatlabError(f"{fn}: LeadingDelimitersRule must be 'keep', 'ignore' or 'error'.")
        self.join = join
        self.layout = Layout(lines, self.delims, join, lead)
        rows = self.layout.rows
        header = None
        for key in ("NumHeaderLines", "HeaderLines"):
            if key in nv:
                header = to_int(nv[key], key)
                if header < 0:
                    raise MatlabError(f"{fn}: {key} must not be negative.")
        want_names = _flag(nv["ReadVariableNames"]) if "ReadVariableNames" in nv else names_default
        self.missing_words = set(text_items(nv["TreatAsMissing"], "TreatAsMissing")) if "TreatAsMissing" in nv \
            else set()
        if "TreatAsEmpty" in nv:
            self.missing_words |= set(text_items(nv["TreatAsEmpty"], "TreatAsEmpty"))
        self.trim = _flag(nv["TrimNonNumeric"]) if "TrimNonNumeric" in nv else False
        start_scan = header or 0
        width = self.layout.width(start_scan)
        if "ExpectedNumVariables" in nv:
            width = to_int(nv["ExpectedNumVariables"], "ExpectedNumVariables")
        names_line, first = detect_layout(rows, width, self.decimal, self.thousands, want_names, header)
        col0, col1 = 0, None
        last = None
        if "Range" in nv:
            r1, c1, r2, c2 = _range_bounds(nv["Range"], fn)
            col0, col1, last = c1, c2, r2
            names_line, first = -1, r1
            use_names = want_names
            if use_names is None and names_default is not False:
                row = rows[r1][c1:(None if c2 is None else c2 + 1)] if r1 < len(rows) and rows[r1] else []
                use_names = bool(row) and _is_texty(row, self.decimal, self.thousands)
            if use_names:
                names_line, first = r1, r1 + 1
        if "VariableNamesLine" in nv:
            n = to_int(nv["VariableNamesLine"], "VariableNamesLine")
            names_line = n - 1
            if n == 0:
                names_line = -1
            elif "DataLines" not in nv and first <= names_line:
                first = names_line + 1
        if "DataLines" in nv:
            a = np.asarray(nv["DataLines"], dtype=float).reshape(-1)
            first = int(a[0]) - 1
            if a.size > 1 and np.isfinite(a[1]):
                last = int(a[1]) - 1
        self.names_line = names_line
        self.first = first
        skip_empty = True
        if "EmptyLineRule" in nv:
            rule = text(nv["EmptyLineRule"]).lower()
            if rule not in _RULES["EmptyLineRule"]:
                raise MatlabError(f"{fn}: EmptyLineRule must be 'skip' or 'read'.")
            skip_empty = rule == "skip"
        stop = len(rows) if last is None else min(last + 1, len(rows))
        body = []
        for r in rows[first:stop]:
            if r is None:
                if not skip_empty:
                    body.append([])
                continue
            body.append(r)
        width = max([len(r) for r in body] + [len(rows[names_line]) if 0 <= names_line < len(rows)
                                              and rows[names_line] else 0] + [0])
        extra = text(nv["ExtraColumnsRule"]).lower() if "ExtraColumnsRule" in nv else "addvars"
        if extra not in _RULES["ExtraColumnsRule"]:
            raise MatlabError(f"{fn}: ExtraColumnsRule must be 'addvars', 'ignore', 'wrap' or 'error'.")
        if "ExpectedNumVariables" in nv:
            want = to_int(nv["ExpectedNumVariables"], "ExpectedNumVariables")
            if width > want:
                if extra == "error":
                    raise MatlabError(f"{fn}: a line has more than the expected {want} fields.")
                if extra == "ignore":
                    width = want
                elif extra == "wrap":
                    wrapped = []
                    for r in body:
                        for k in range(0, max(len(r), 1), want):
                            wrapped.append(r[k:k + want])
                    body = wrapped
                    width = want
            else:
                width = want
        # a delimiter at the end of every line does not start another column
        trailing = text(nv["TrailingDelimitersRule"]).lower() if "TrailingDelimitersRule" in nv else "ignore"
        if trailing not in _RULES["TrailingDelimitersRule"]:
            raise MatlabError(f"{fn}: TrailingDelimitersRule must be 'keep', 'ignore' or 'error'.")
        if trailing != "keep" and width > 1 and body and self.delims is not None and \
                all(len(r) < width or r[width - 1] == "" for r in body) and \
                any(len(r) == width for r in body) and "ExpectedNumVariables" not in nv:
            head = rows[names_line] if 0 <= names_line < len(rows) and rows[names_line] else []
            if len(head) < width or head[width - 1].strip() == "":
                if trailing == "error":
                    raise MatlabError(f"{fn}: lines end with a delimiter.")
                width -= 1
        if col1 is not None:
            width = min(width, col1 + 1)
        self.col0 = col0
        self.width = max(width - col0, 0)
        self.body = [(r[col0:col0 + self.width] + [""] * self.width)[:self.width] for r in body]
        head = rows[names_line] if 0 <= names_line < len(rows) and rows[names_line] else None
        self.header = None if head is None else (head[col0:col0 + self.width] + [""] * self.width)[:self.width]

    def column(self, k: int) -> list[str]:
        return [r[k] for r in self.body]

    def defaults(self) -> dict:
        return {"decimal": self.decimal, "thousands": self.thousands, "missing": self.missing_words,
                "trim": self.trim}


def detect(interp, fname: str, nv: dict, fn: str = "detectImportOptions") -> ImportOptions:
    p = Parsed(interp, fname, nv, fn, None)
    o = ImportOptions()
    o.file = fname
    o.delimiter = p.delims
    o.range = nv.get("Range")
    rule = "modify"
    if "VariableNamingRule" in nv:
        rule = text(nv["VariableNamingRule"]).lower()
        if rule not in ("modify", "preserve"):
            raise MatlabError(f"{fn}: VariableNamingRule must be 'modify' or 'preserve'.")
    if "PreserveVariableNames" in nv:
        rule = "preserve" if _flag(nv["PreserveVariableNames"]) else "modify"
    o.props["VariableNamingRule"] = rule
    o.names = valid_names(p.header if p.header is not None else [""] * p.width, rule)
    o.text_type = _choice(nv, "TextType", ("char", "string"), "char", fn)
    o.datetime_type = _choice(nv, "DatetimeType", ("datetime", "text"), "datetime", fn)
    duration_type = _choice(nv, "DurationType", ("duration", "text"), "duration", fn)
    o.types = []
    for k in range(p.width):
        kind = column_kind(p.column(k), p.decimal, p.thousands, p.missing_words, p.trim)
        if kind == "datetime" and o.datetime_type == "text":
            kind = "char"
        if kind == "duration" and duration_type == "text":
            kind = "char"
        if kind == "char" and o.text_type == "string":
            kind = "string"
        o.types.append(kind)
    o.var_opts = []
    for n, t in zip(o.names, o.types):
        vo: dict = {"Name": n, "Type": t}
        if p.missing_words:
            vo["TreatAsMissing"] = sorted(p.missing_words)
        if t == "double":
            if p.decimal != ".":
                vo["DecimalSeparator"] = p.decimal
            if p.thousands:
                vo["ThousandsSeparator"] = p.thousands
            if p.trim:
                vo["TrimNonNumeric"] = True
        o.var_opts.append(vo)
    o.props["DataLines"] = np.array([[float(p.first + 1), np.inf]])
    o.props["VariableNamesLine"] = float(p.names_line + 1)
    for key in ("MissingRule", "ImportErrorRule", "EmptyLineRule", "ExtraColumnsRule", "ConsecutiveDelimitersRule",
                "LeadingDelimitersRule", "TrailingDelimitersRule"):
        if key in nv:
            v = text(nv[key]).lower()
            if v not in _RULES[key]:
                raise MatlabError(f"{fn}: {key} must be one of: {', '.join(_RULES[key])}.")
            o.props[key] = v
    if p.join:
        o.props["ConsecutiveDelimitersRule"] = "join"
    if p.comment_style:
        o.props["CommentStyle"] = p.comment_style
    if "Encoding" in nv:
        o.props["Encoding"] = text(nv["Encoding"])
    if "Whitespace" in nv:
        o.props["Whitespace"] = text(nv["Whitespace"])
    if "LineEnding" in nv:
        o.props["LineEnding"] = [x.replace("\\n", "\n").replace("\\r", "\r")
                                 for x in text_items(nv["LineEnding"], "LineEnding")]
    if "RowNamesColumn" in nv:
        o.props["RowNamesColumn"] = float(to_int(nv["RowNamesColumn"], "RowNamesColumn"))
    if "ReadRowNames" in nv and _flag(nv["ReadRowNames"]):
        o.props["RowNamesColumn"] = 1.0
    if "VariableNames" in nv:
        o = o.set_field(interp, "VariableNames", nv["VariableNames"])
    if "VariableTypes" in nv:
        o = o.set_field(interp, "VariableTypes", nv["VariableTypes"])
    if "SelectedVariableNames" in nv:
        o = o.set_field(interp, "SelectedVariableNames", nv["SelectedVariableNames"])
    return o


def _choice(nv: dict, key: str, allowed: tuple, default: str, fn: str) -> str:
    if key not in nv:
        return default
    v = text(nv[key]).lower()
    if v not in allowed:
        raise MatlabError(f"{fn}: {key} must be one of: {', '.join(allowed)}.")
    return v


def _options_nv(o: ImportOptions) -> dict:
    """The name-value settings that reproduce how the options object reads its file."""
    nv: dict = {"Delimiter": Cell.from_list([char(d) for d in (o.delimiter or [" "])]),
                "DataLines": o.props["DataLines"],
                "VariableNamesLine": np.array([[o.props["VariableNamesLine"]]]),
                "EmptyLineRule": char(o.props["EmptyLineRule"]),
                "ConsecutiveDelimitersRule": char(o.props["ConsecutiveDelimitersRule"]),
                "LeadingDelimitersRule": char(o.props["LeadingDelimitersRule"]),
                "TrailingDelimitersRule": char(o.props["TrailingDelimitersRule"]),
                "ExtraColumnsRule": char(o.props["ExtraColumnsRule"]),
                "ExpectedNumVariables": np.array([[float(len(o.names))]]),
                "Encoding": char(o.props["Encoding"])}
    if o.props["CommentStyle"]:
        nv["CommentStyle"] = Cell.from_list([char(x) for x in o.props["CommentStyle"]])
    if o.props["LineEnding"] != ["\n", "\r", "\r\n"]:
        nv["LineEnding"] = Cell.from_list([char(x) for x in o.props["LineEnding"]])
    if o.range is not None:
        nv["Range"] = o.range
    return nv


def read_columns(interp, fname: str, o: ImportOptions, fn: str, extra_nv: dict | None = None):
    """(names, columns, row names) of the selected variables, read as the options say."""
    nv = _options_nv(o)
    nv.update(extra_nv or {})
    p = Parsed(interp, fname, nv, fn, o.props["VariableNamesLine"] > 0)
    defaults = {"decimal": ".", "thousands": "", "missing": set(), "trim": False}
    cols: list = []
    bads: list = []
    idx = o.selected_positions()
    row_col = int(o.props["RowNamesColumn"])
    for k in idx:
        items = p.column(k) if k < p.width else [""] * len(p.body)
        col, bad = convert_column(items, o.types[k], o.var_opts[k], defaults, o.names[k])
        cols.append(col)
        bads.append(bad)
    names = [o.names[k] for k in idx]
    if p.width > len(o.names) and o.props["ExtraColumnsRule"] == "addvars" and o.selected is None:
        for k in range(len(o.names), p.width):  # columns the options did not expect
            col, bad = convert_column(p.column(k), "char", {}, defaults, "")
            cols.append(col)
            bads.append(np.zeros(len(p.body), dtype=bool))
            names.append(f"ExtraVar{k - len(o.names) + 1}")
    row_names = None
    if row_col:
        if row_col > p.width:
            raise MatlabError(f"{fn}: RowNamesColumn {row_col} is beyond the last column.")
        row_names = [s.strip() for s in p.column(row_col - 1)]
        if row_col - 1 in idx:
            j = idx.index(row_col - 1)
            del names[j], cols[j], bads[j]
    nrows = len(p.body)
    keep_rows = np.ones(nrows, dtype=bool)
    keep_cols = [True] * len(cols)
    for rule_name in ("MissingRule", "ImportErrorRule"):
        rule = o.props[rule_name]
        if rule == "fill":
            continue
        for j, bad in enumerate(bads):
            if not bad.any():
                continue
            if rule == "error":
                line = int(np.flatnonzero(bad)[0]) + p.first + 1
                raise MatlabError(f"{fn}: unable to import the field in line {line}, variable '{names[j]}': it is "
                                  f"missing or not a {o.types[o.names.index(names[j])]} value "
                                  f"({rule_name} is 'error').", "MATLAB:textio:io:ImportError")
            if rule == "omitrow":
                keep_rows &= ~bad
            elif rule == "omitvar":
                keep_cols[j] = False
    if not keep_rows.all():
        from .table import _rows
        pos = np.flatnonzero(keep_rows)
        cols = [_rows(c, pos) for c in cols]
        if row_names is not None:
            row_names = [row_names[i] for i in pos]
    names = [n for n, k in zip(names, keep_cols) if k]
    cols = [c for c, k in zip(cols, keep_cols) if k]
    return names, cols, row_names


_FORMAT_TYPES = {"f": "double", "n": "double", "d": "double", "u": "double", "s": "char", "q": "char",
                 "D": "datetime", "C": "categorical", "T": "duration"}


def read_table(interp, args: list, fn: str = "readtable"):
    from .table import TableObject
    fname = text(args[0], "file name")
    rest = list(args[1:])
    opts = None
    if rest and isinstance(rest[0], ImportOptions):
        opts = rest.pop(0)
    nv = name_values(rest, KNOWN, fn)
    extra: dict = {}
    if opts is None:
        opts = detect(interp, fname, {k: v for k, v in nv.items() if k != "Format"}, fn)
        if "Format" in nv:
            specs = re.findall(r"%(\*?)(?:\d*)(?:\{[^}]*\})?([a-zA-Z])", text(nv["Format"]))
            if len(specs) != len(opts.names):
                raise MatlabError(f"{fn}: the Format has {len(specs)} conversions but the file has "
                                  f"{len(opts.names)} columns.")
            keep = []
            for k, (skip, code) in enumerate(specs):
                if code not in _FORMAT_TYPES:
                    raise MatlabError(f"{fn}: the conversion '%{code}' in Format is not supported.")
                opts.types[k] = _FORMAT_TYPES[code]
                opts.var_opts[k]["Type"] = opts.types[k]
                if not skip:
                    keep.append(opts.names[k])
            opts.selected = keep
    else:
        opts = opts.copy()
        allowed = {"ReadVariableNames", "ReadRowNames", "TextType", "DatetimeType", "VariableNamingRule",
                   "PreserveVariableNames", "MissingRule", "ImportErrorRule", "Encoding", "DateLocale"}
        for key in nv:
            if key not in allowed:
                raise MatlabError(f"{fn}: the option '{key}' cannot be combined with an import options object; "
                                  "set it on the object instead.")
        if "ReadVariableNames" in nv and not _flag(nv["ReadVariableNames"]):
            if opts.props["VariableNamesLine"] > 0:
                first = float(opts.props["DataLines"][0, 0])
                opts.props["DataLines"] = np.array([[min(first, opts.props["VariableNamesLine"]), np.inf]])
            opts.props["VariableNamesLine"] = 0.0
        if "ReadRowNames" in nv and _flag(nv["ReadRowNames"]):
            opts.props["RowNamesColumn"] = 1.0
        for key in ("MissingRule", "ImportErrorRule"):
            if key in nv:
                opts = opts.set_field(interp, key, nv[key])
        if "TextType" in nv:
            want = _choice(nv, "TextType", ("char", "string"), "char", fn)
            opts.types = [want if t in ("char", "string") else t for t in opts.types]
    names, cols, row_names = read_columns(interp, fname, opts, fn, extra)
    if row_names is not None and len(set(row_names)) != len(row_names):
        raise MatlabError(f"{fn}: the row names read from the file are not unique.")
    return TableObject(names, cols, row_names=row_names)


def read_matrix(interp, args: list, fn: str = "readmatrix"):
    fname = text(args[0], "file name")
    rest = list(args[1:])
    opts = None
    if rest and isinstance(rest[0], ImportOptions):
        opts = rest.pop(0)
    nv = name_values(rest, KNOWN, fn)
    out_type = "double"
    if "OutputType" in nv:
        out_type = text(nv["OutputType"])
        if out_type not in _TYPES:
            raise MatlabError(f"{fn}: OutputType '{out_type}' is not supported. Types are: {', '.join(_TYPES)}.")
    if opts is not None:
        o = opts.copy()
        if "OutputType" in nv:
            o.types = [out_type] * len(o.types)
        names, cols, _ = read_columns(interp, fname, o, fn)
        return _join_columns(cols, fn)
    p = Parsed(interp, fname, {k: v for k, v in nv.items() if k != "OutputType"}, fn,
               None if "Range" not in nv else False)
    if out_type in ("char", "string", "datetime", "duration", "categorical", "logical"):
        cols = [convert_column(p.column(k), out_type, {}, p.defaults(), "")[0] for k in range(p.width)]
        return _join_columns(cols, fn)
    data = np.full((len(p.body), p.width), np.nan, dtype=complex)
    for i, r in enumerate(p.body):
        for j, f in enumerate(r):
            t = f.strip()
            if t == "" or t in p.missing_words:
                continue
            v = parse_number(t, p.decimal, p.thousands, p.trim)
            if v is not None:
                data[i, j] = v
    out: np.ndarray = data.real if not np.any(np.nan_to_num(data.imag) != 0) else data
    if out_type == "single":
        out = out.astype(np.complex64 if out.dtype.kind == "c" else np.float32)
    elif out_type != "double":
        from ..values import INT_TYPES, saturate_int
        out = saturate_int(np.real(out), INT_TYPES[out_type])
    return np.ascontiguousarray(out)


def _join_columns(cols: list, fn: str) -> Any:
    if not cols:
        return np.zeros((0, 0))
    first = cols[0]
    if isinstance(first, np.ndarray):
        if not all(isinstance(c, np.ndarray) for c in cols):
            raise MatlabError(f"{fn}: the variables have different types and cannot be joined into one matrix; "
                              "use readtable or readcell.")
        return np.hstack([c.astype(np.result_type(*[x.dtype for x in cols])) for c in cols])
    if isinstance(first, Cell):
        if not all(isinstance(c, Cell) for c in cols):
            raise MatlabError(f"{fn}: the variables have different types; use readtable or readcell.")
        return Cell(np.hstack([c.data for c in cols]))
    if isinstance(first, StringArray):
        if not all(isinstance(c, StringArray) for c in cols):
            raise MatlabError(f"{fn}: the variables have different types; use readtable or readcell.")
        return StringArray(np.hstack([c.data for c in cols]))
    if all(type(c) is type(first) for c in cols) and hasattr(first, "like"):
        return first.like(np.hstack([c.data for c in cols]))
    raise MatlabError(f"{fn}: the variables have different types; use readtable or readcell.")


def read_cell(interp, args: list, fn: str = "readcell"):
    from .datetimes import DatetimeArray, DurationArray
    fname = text(args[0], "file name")
    rest = list(args[1:])
    opts = None
    if rest and isinstance(rest[0], ImportOptions):
        opts = rest.pop(0)
    nv = name_values(rest, KNOWN, fn)
    if opts is not None:
        names, cols, _ = read_columns(interp, fname, opts, fn)
        n = len(cols[0].data if hasattr(cols[0], "data") else cols[0]) if cols else 0
        data = np.empty((n, len(cols)), dtype=object)
        from .table import _rows
        for j, c in enumerate(cols):
            for i in range(n):
                v = _rows(c, np.array([i]))
                data[i, j] = v.data[0, 0] if isinstance(v, Cell) else \
                    (string_result(v.data) if isinstance(v, StringArray) else v)
        return Cell(data)
    text_type = _choice(nv, "TextType", ("char", "string"), "char", fn)
    dt_type = _choice(nv, "DatetimeType", ("datetime", "text"), "datetime", fn)
    du_type = _choice(nv, "DurationType", ("duration", "text"), "duration", fn)
    p = Parsed(interp, fname, {k: v for k, v in nv.items() if k not in ("TextType", "DatetimeType", "DurationType")},
               fn, False)
    missing = np.empty((1, 1), dtype=object)
    missing[0, 0] = None
    data = np.empty((len(p.body), p.width), dtype=object)
    for i, r in enumerate(p.body):
        for j, f in enumerate(r):
            t = f.strip()
            if t == "" or t in p.missing_words:
                data[i, j] = StringArray(missing.copy())
                continue
            v = parse_number(t, p.decimal, p.thousands, p.trim)
            if v is not None:
                data[i, j] = np.array([[v]])
                continue
            if du_type == "duration" and _DURATION_RE.match(t):
                d = parse_durations([t])
                data[i, j] = DurationArray(d.reshape(1, 1), None)
                continue
            if dt_type == "datetime":
                res = try_datetimes([t])
                if res is not None:
                    data[i, j] = DatetimeArray(res[0].reshape(1, 1), None, "")
                    continue
            data[i, j] = MString(f) if text_type == "string" else char(f)
    return Cell(data)


# -- datastore ---------------------------------------------------------------------------------------

class TabularTextDatastore(MObject):
    """A collection of delimited text files read in blocks of rows."""

    is_handle = True
    _METHODS = {"read", "readall", "hasdata", "reset", "preview", "progress", "numpartitions", "partition",
                "subset", "shuffle", "transform", "combine", "tall", "isPartitionable", "isShuffleable"}

    def __init__(self, interp, files: list[str], nv: dict):
        self.files = files
        self.read_size: Any = 20000
        self.nv = nv
        self.opts = detect(interp, files[0], nv, "tabularTextDatastore")
        self.file_index = 0
        self.row = 0
        self.cache: tuple | None = None

    def class_name(self) -> str:
        return "matlab.io.datastore.TabularTextDatastore"

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "matlab.io.Datastore", "handle")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def _table(self, interp, k: int):
        if self.cache is None or self.cache[0] != k:
            o = self.opts.copy()
            o.file = self.files[k]
            names, cols, _ = read_columns(interp, self.files[k], o, "datastore")
            from .table import TableObject
            self.cache = (k, TableObject(names, cols))
        return self.cache[1]

    def _slice(self, t, a: int, b: int):
        from .table import TableObject, _rows
        pos = np.arange(a, b)
        return TableObject(list(t.names), [_rows(c, pos) for c in t.columns])

    def hasdata(self, interp) -> bool:
        while self.file_index < len(self.files):
            if self.row < self._table(interp, self.file_index).nrows:
                return True
            self.file_index += 1
            self.row = 0
        return False

    def read(self, interp):
        if not self.hasdata(interp):
            raise MatlabError("No more data to read. Use reset to read from the start, or check hasdata first.",
                              "MATLAB:datastoreio:splittabledatastore:noMoreData")
        t = self._table(interp, self.file_index)
        if self.read_size == "file":
            a, b = self.row, t.nrows
        else:
            a, b = self.row, min(self.row + int(self.read_size), t.nrows)
        self.row = b
        info = Struct.scalar({"Filename": char(os.path.abspath(self.files[self.file_index])),
                              "FileSize": np.array([[float(os.path.getsize(self.files[self.file_index]))]]),
                              "Offset": np.array([[float(a)]]), "NumCharactersRead": np.array([[float(b - a)]])})
        return self._slice(t, a, b), info

    def readall(self, interp):
        from .table import TableObject
        tables = [self._table(interp, k) for k in range(len(self.files))]
        self.cache = None
        if len(tables) == 1:
            return tables[0]
        return TableObject.concat(tables, 0)

    def get_field(self, interp, name: str) -> Any:
        o = self.opts
        if name == "Files":
            return Cell.from_list([char(os.path.abspath(f)) for f in self.files], column=True)
        if name == "ReadSize":
            return char("file") if self.read_size == "file" else np.array([[float(self.read_size)]])
        if name in ("VariableNames", "SelectedVariableNames", "Delimiter"):
            return o.get_field(interp, name)
        if name == "SelectedFormats" or name == "TextscanFormats":
            code = {"double": "%f", "char": "%q", "string": "%q", "datetime": "%D", "duration": "%T",
                    "categorical": "%C"}
            idx = o.selected_positions() if name == "SelectedFormats" else range(len(o.names))
            return Cell.from_list([char(code.get(o.types[k], "%f")) for k in idx])
        if name == "NumHeaderLines":
            return np.array([[max(float(o.props["DataLines"][0, 0]) - 1 - (o.props["VariableNamesLine"] > 0), 0.0)]])
        if name == "ReadVariableNames":
            return np.array([[o.props["VariableNamesLine"] > 0]])
        if name in ("TreatAsMissing",):
            return Cell.from_list([char(x) for x in sorted(o.var_opts[0].get("TreatAsMissing", []))]) \
                if o.var_opts else Cell.empty(0, 0)
        if name == "MissingValue":
            return np.array([[np.nan]])
        if name == "TextType":
            return char(o.text_type)
        if name == "DatetimeType":
            return char(o.datetime_type)
        if name in self._METHODS:
            return super().get_field(interp, name)
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def set_field(self, interp, name: str, value: Any) -> "TabularTextDatastore":
        if name == "ReadSize":
            if is_text(value):
                if text(value).lower() != "file":
                    raise MatlabError("ReadSize must be a positive number of rows or 'file'.")
                self.read_size = "file"
            else:
                n = to_int(value, "ReadSize")
                if n < 1:
                    raise MatlabError("ReadSize must be a positive number of rows or 'file'.")
                self.read_size = n
        elif name in ("SelectedVariableNames", "VariableNames"):
            self.opts = self.opts.set_field(interp, name, value)
            self.cache = None
        elif name == "SelectedFormats":
            fmts = text_items(value, "SelectedFormats")
            idx = self.opts.selected_positions()
            if len(fmts) != len(idx):
                raise MatlabError("SelectedFormats needs one format per selected variable.")
            for k, f in zip(idx, fmts):
                code = f.strip().lstrip("%")[-1:]
                if code not in _FORMAT_TYPES:
                    raise MatlabError(f"The format '{f}' is not supported.")
                self.opts.types[k] = _FORMAT_TYPES[code]
            self.cache = None
        else:
            raise MatlabError(f"Cannot set property '{name}' of a datastore.")
        return self

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name == "read":
            t, info = self.read(interp)
            return [t, info][:max(nargout, 1)]
        if name == "readall":
            return [self.readall(interp)]
        if name == "hasdata":
            return [np.array([[self.hasdata(interp)]])]
        if name == "reset":
            self.file_index, self.row = 0, 0
            return []
        if name == "preview":
            t = self._table(interp, 0)
            return [self._slice(t, 0, min(8, t.nrows))]
        if name == "progress":
            total = sum(self._table(interp, k).nrows for k in range(len(self.files))) or 1
            done = sum(self._table(interp, k).nrows for k in range(min(self.file_index, len(self.files)))) + self.row
            return [np.array([[done / total]])]
        if name == "numpartitions":
            return [np.array([[float(len(self.files))]])]
        if name in ("isPartitionable", "isShuffleable"):
            return [np.array([[True]])]
        if name in ("partition", "subset"):
            if name == "partition":
                if len(rest) != 2:
                    raise MatlabError("partition(ds, n, index): the number of parts and the part to return.")
                n, k = to_int(rest[0], "n"), to_int(rest[1], "index")
                if not 1 <= k <= n:
                    raise MatlabError("partition: index must be between 1 and n.")
                edges = np.linspace(0, len(self.files), n + 1).round().astype(int)
                files = self.files[edges[k - 1]:edges[k]]
            else:
                idx = [int(x) - 1 for x in np.asarray(rest[0]).flatten()] if rest else []
                if rest and np.asarray(rest[0]).dtype == bool:
                    idx = [i for i, f in enumerate(np.asarray(rest[0]).flatten()) if f]
                files = [self.files[i] for i in idx]
            if not files:
                raise MatlabError(f"{name}: the selection holds no files.")
            other = TabularTextDatastore(interp, files, self.nv)
            other.opts = self.opts.copy()
            other.read_size = self.read_size
            return [other]
        if name == "shuffle":
            from .core import _rng
            order = _rng(interp).permutation(len(self.files))
            other = TabularTextDatastore(interp, [self.files[i] for i in order], self.nv)
            other.opts = self.opts.copy()
            other.read_size = self.read_size
            return [other]
        if name == "transform":
            from .imagedatastore import _make_transform
            return [_make_transform(interp, self, rest)]
        if name == "combine":
            from .imagedatastore import CombinedDatastore, _fresh, _require_ds
            return [CombinedDatastore([_fresh(interp, _require_ds(a, "combine")) for a in args])]
        raise MatlabError(f"The datastore function '{name}' is not supported.")

    def display_lines(self) -> list[str]:
        names = [self.opts.names[k] for k in self.opts.selected_positions()]
        shown = "'" + "', '".join(os.path.abspath(f) for f in self.files[:3]) + "'" + \
            (" ..." if len(self.files) > 3 else "")
        return ["  TabularTextDatastore with properties:", "",
                f"                      Files: {{{shown}}}",
                f"                   ReadSize: {self.read_size} rows" if self.read_size != "file" else
                "                   ReadSize: 'file'",
                f"              VariableNames: {{{', '.join(repr(n) for n in self.opts.names)}}}",
                f"      SelectedVariableNames: {{{', '.join(repr(n) for n in names)}}}"]


def datastore_files(interp, location: Any, extensions: tuple | None, include_sub: bool) -> list[str]:
    import glob
    from .fileformats import find_file
    out: list[str] = []
    for loc in text_items(location, "The location"):
        loc = os.path.expanduser(loc)
        if os.path.isdir(loc):
            pattern = os.path.join(loc, "**", "*") if include_sub else os.path.join(loc, "*")
            found = sorted(f for f in glob.glob(pattern, recursive=include_sub) if os.path.isfile(f))
        elif any(c in loc for c in "*?["):
            found = sorted(f for f in glob.glob(loc, recursive=True) if os.path.isfile(f))
        else:
            p = find_file(interp, loc)
            if p is None:
                raise MatlabError(f"Cannot find files or folders matching: '{loc}'.",
                                  "MATLAB:datastoreio:pathlookup:fileNotFound")
            found = [p]
        if extensions is not None:
            found = [f for f in found if os.path.splitext(f)[1].lower() in extensions]
        out.extend(found)
    if not out:
        raise MatlabError("Cannot find files or folders matching the location.",
                          "MATLAB:datastoreio:pathlookup:fileNotFound")
    return out


# -- textscan --------------------------------------------------------------------------------------

_TS_SPEC = re.compile(r"%(\*?)(\d*)(?:\.(\d+))?(?:\{([^}]*)\})?(f|d|u|n|s|q|c|D|T|C|\[\^?\]?[^\]]*\])(8|16|32|64)?")
_TS_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*(?:[eEdD][+-]?\d+)?|\.\d+(?:[eEdD][+-]?\d+)?|inf(?:inity)?|nan)",
                        re.IGNORECASE)
_TS_IMAG = re.compile(r"[+-](?:\d+\.?\d*(?:[eEdD][+-]?\d+)?|\.\d+(?:[eEdD][+-]?\d+)?)[ij]")
_TS_INT = re.compile(r"[+-]?\d+")
_TS_KNOWN = _known("Delimiter", "HeaderLines", "EndOfLine", "MultipleDelimsAsOne", "CollectOutput", "Whitespace",
                   "CommentStyle", "ReturnOnError", "EmptyValue", "TextType", "TreatAsEmpty", "DateLocale",
                   "ExpChars", "BufSize", unsupported=("DatetimeType",))


class _Stop(Exception):
    """Raised inside the scanner when a field does not fit its conversion."""


class _Scanner:
    def __init__(self, s: str, fmt: str, nv: dict):
        from .strings import process_escapes
        self.s = s
        self.n = len(s)
        fmt = process_escapes(fmt).replace("%%", "\0")
        self.specs = []
        pos = 0
        for m in _TS_SPEC.finditer(fmt):
            kind = m.group(5)
            chars, negate = "", False
            if kind[0] == "[":
                body = kind[1:-1]
                negate = body.startswith("^")
                chars = body[1:] if negate else body
                kind = "["
            self.specs.append({"kind": kind, "skip": bool(m.group(1)),
                               "width": int(m.group(2)) if m.group(2) else None, "datefmt": m.group(4),
                               "bits": m.group(6) or "", "chars": chars, "negate": negate,
                               "literal": fmt[pos:m.start()].strip().replace("\0", "%")})
            pos = m.end()
        if not self.specs:
            raise MatlabError("textscan: the format must contain at least one conversion such as %f or %s.")
        self.tail = fmt[pos:].strip().replace("\0", "%")
        self.delims: list[str] | None = None
        if "Delimiter" in nv:
            d = nv["Delimiter"]
            items = [to_str(x) for x in d.values_f()] if isinstance(d, Cell) else text_items(d, "Delimiter")
            self.delims = [process_escapes(x) for x in items if x != ""] or None
            if self.delims:
                self.delims.sort(key=len, reverse=True)
        self.ws = process_escapes(text(nv["Whitespace"])) if "Whitespace" in nv else " \b\t"
        if "EndOfLine" in nv:
            eol = process_escapes(text(nv["EndOfLine"]))
            if eol == "":
                raise MatlabError("textscan: EndOfLine must not be empty.")
            self.eols = [eol]
            # the usual line breaks are plain white space when another end-of-line sequence is set
            self.ws += "".join(c for c in "\r\n" if c not in eol)
        else:
            self.eols = ["\r\n", "\n", "\r"]
        self.merge = "MultipleDelimsAsOne" in nv and _flag(nv["MultipleDelimsAsOne"])
        self.empty = float(np.asarray(nv["EmptyValue"], dtype=float).flat[0]) if "EmptyValue" in nv else float("nan")
        self.treat_empty = set(text_items(nv["TreatAsEmpty"], "TreatAsEmpty")) if "TreatAsEmpty" in nv else set()
        self.comment: list[str] = []
        if "CommentStyle" in nv:
            self.comment = [process_escapes(x) for x in text_items(nv["CommentStyle"], "CommentStyle")]
            if len(self.comment) not in (1, 2) or not all(self.comment):
                raise MatlabError("textscan: CommentStyle must be one marker or a pair {open, close}.")
        self.return_on_error = _flag(nv["ReturnOnError"]) if "ReturnOnError" in nv else True
        self.columns: list[list] = [[] for _ in self.specs]
        self.p = 0

    # positions ----------------------------------------------------------------------------------
    def eol_at(self, p: int) -> int:
        for e in self.eols:
            if self.s.startswith(e, p):
                return len(e)
        return 0

    def delim_at(self, p: int) -> str | None:
        if self.delims is None:
            return None
        for d in self.delims:
            if self.s.startswith(d, p):
                return d
        return None

    def skip(self, p: int, lines: bool) -> int:
        """Skip white space and comments, and line ends too when ``lines`` is set."""
        s, n = self.s, self.n
        while p < n:
            ch = s[p]
            if ch in self.ws and self.delim_at(p) is None:
                p += 1
                continue
            if self.comment and s.startswith(self.comment[0], p):
                if len(self.comment) == 2:
                    q = s.find(self.comment[1], p + len(self.comment[0]))
                    p = n if q < 0 else q + len(self.comment[1])
                else:
                    while p < n and not self.eol_at(p):
                        p += 1
                continue
            if lines:
                e = self.eol_at(p)
                if e:
                    p += e
                    continue
            break
        return p

    def field_end(self, p: int) -> int:
        """Where the field starting at p ends: a delimiter, a line end, or (without delimiters) white space."""
        s, n = self.s, self.n
        q = p
        while q < n:
            if self.eol_at(q) or self.delim_at(q) is not None:
                break
            if self.delims is None and s[q] in self.ws:
                break
            if self.comment and s.startswith(self.comment[0], q):
                break
            q += 1
        return q

    def skip_header(self, count: int) -> None:
        p = 0
        for _ in range(count):
            while p < self.n and not self.eol_at(p):
                p += 1
            p += self.eol_at(p)
        self.p = p

    # conversions --------------------------------------------------------------------------------
    def empty_here(self, p: int) -> bool:
        return self.delims is not None and (p >= self.n or self.delim_at(p) is not None or bool(self.eol_at(p)))

    def convert(self, spec: dict, p: int) -> tuple[Any, int]:
        s, n = self.s, self.n
        kind, width = spec["kind"], spec["width"]
        limit = n if width is None else min(n, p + width)
        if kind in "fndu":
            if self.empty_here(p):
                return self.empty, p
            if self.treat_empty:
                end = self.field_end(p)
                if s[p:end].strip() in self.treat_empty:
                    return self.empty, end
            m = (_TS_INT if kind in "du" else _TS_NUMBER).match(s, p, limit)
            if not m:
                raise _Stop()
            end = m.end()
            t = m.group(0).lower().replace("d", "e").replace("infinity", "inf") if kind in "fn" else m.group(0)
            value: Any = float(t)
            if kind in "fn":
                mi = _TS_IMAG.match(s, end, limit)
                if mi:
                    value = complex(value, float(mi.group(0)[:-1].lower().replace("d", "e")))
                    end = mi.end()
            return value, end
        if kind == "c":
            count = width or 1
            if p + count > n:
                raise _Stop()
            return s[p:p + count], p + count
        if kind == "[":
            q = p
            while q < limit and ((s[q] in spec["chars"]) != spec["negate"]):
                q += 1
            if q == p:
                raise _Stop()
            return s[p:q], q
        if kind == "q" and p < n and s[p] == '"':
            q = p + 1
            buf = []
            while q < n:
                if s[q] == '"':
                    if q + 1 < n and s[q + 1] == '"':
                        buf.append('"')
                        q += 2
                        continue
                    break
                buf.append(s[q])
                q += 1
            return "".join(buf), min(q + 1, n)
        # %s, %q without quotes, %D, %T, %C: one field of text
        end = min(self.field_end(p), limit)
        if kind == "D" and self.delims is None and spec["datefmt"] and end < n:
            for _ in range(spec["datefmt"].strip().count(" ")):  # a date format with spaces spans several words
                q = end
                while q < n and s[q] in self.ws:
                    q += 1
                end = self.field_end(q)
        value = s[p:end]
        if self.delims is not None:
            value = value.rstrip(self.ws)
        if value == "" and self.delims is None:
            raise _Stop()
        return value, end

    def run(self, repeat: int | None) -> None:
        s, n = self.s, self.n
        last = len(self.specs) - 1
        records = 0
        while repeat is None or records < repeat:
            p = self.skip(self.p, True)
            if p >= n:
                self.p = p
                break
            start = p
            ended = False  # the line ended before the format did: the remaining fields are empty
            after_delim = False
            try:
                for j, spec in enumerate(self.specs):
                    col = self.columns[j]
                    if ended:
                        if not spec["skip"]:
                            col.append(self.empty if spec["kind"] in "fndu" else "")
                        continue
                    p = self.skip(p, self.delims is None)
                    if p >= n and not after_delim:
                        raise _Stop()  # the text ended inside a record: nothing more to read
                    if spec["literal"]:
                        if not s.startswith(spec["literal"], p):
                            raise _Stop()
                        p = self.skip(p + len(spec["literal"]), False)
                    value, p = self.convert(spec, p)
                    if not spec["skip"]:
                        col.append(value)
                    self.p = p
                    q = self.skip(p, False)
                    d = self.delim_at(q)
                    after_delim = d is not None
                    if d is not None:
                        p = q + len(d)
                        if self.merge:
                            while True:
                                q2 = self.skip(p, False)
                                d2 = self.delim_at(q2)
                                if d2 is None:
                                    break
                                p = q2 + len(d2)
                    else:
                        e = self.eol_at(q)
                        if e:
                            p = q + e
                            ended = self.delims is not None and j < last
                        elif self.delims is None or q >= n:
                            p = q
                if self.tail:
                    p = self.skip(p, False)
                    if s.startswith(self.tail, p):
                        p += len(self.tail)
            except _Stop:
                if not self.return_on_error:
                    line = s.count("\n", 0, p) + 1
                    shown = s[p:p + 20].split("\n")[0]
                    raise MatlabError(f"textscan: the text does not match the format at line {line}, field "
                                      f"{j + 1} ('{shown}').", "MATLAB:textscan:handleErrorAndShowInfo") from None
                if self.p < start:
                    self.p = start
                break
            self.p = p
            records += 1


def _ts_column(spec: dict, col: list, as_string: bool) -> Any:
    kind, bits = spec["kind"], spec["bits"]
    if kind in "fn":
        if any(isinstance(x, complex) for x in col):
            return np.array(col, dtype=complex).reshape(-1, 1)
        return np.array(col, dtype=np.float32 if bits == "32" and kind == "f" else float).reshape(-1, 1)
    if kind in "du":
        signed = {"8": np.int8, "16": np.int16, "32": np.int32, "64": np.int64}
        unsigned = {"8": np.uint8, "16": np.uint16, "32": np.uint32, "64": np.uint64}
        dtype = (unsigned if kind == "u" else signed).get(bits, np.uint32 if kind == "u" else np.int32)
        vals = np.array(col, dtype=float)
        info = np.iinfo(dtype)
        vals = np.clip(np.where(np.isnan(vals), 0, np.round(vals)), info.min, info.max)
        return vals.astype(dtype).reshape(-1, 1)
    if kind == "c":
        width = spec["width"] or 1
        if not col:
            return np.zeros((0, width), dtype="<U1")
        return np.array([list(x.ljust(width)[:width]) for x in col], dtype="<U1").reshape(len(col), width)
    if kind == "D":
        from .datetimes import DatetimeArray, parse_texts
        fmt = spec["datefmt"] or None
        secs, _ = parse_texts(col, fmt) if col else (np.zeros(0), False)
        return DatetimeArray(np.asarray(secs, dtype=float).reshape(-1, 1), fmt, "")
    if kind == "T":
        from .datetimes import DurationArray, _parse_duration_text
        return DurationArray(np.array([_parse_duration_text(x) if x.strip() else np.nan for x in col],
                                      dtype=float).reshape(-1, 1), None)
    if kind == "C":
        from .categorical import CategoricalArray
        cats = sorted({x for x in col if x != ""})
        codes = np.array([cats.index(x) + 1.0 if x != "" else np.nan for x in col]).reshape(-1, 1)
        return CategoricalArray(codes, cats, False)
    if as_string:
        d = np.empty((len(col), 1), dtype=object)
        for i, x in enumerate(col):
            d[i, 0] = x
        return StringArray(d)
    return Cell.from_list([char(x) for x in col], column=True)


def textscan(interp, args: list, nargout: int):
    from .system import file_handle
    if not args:
        raise MatlabError("textscan needs a file identifier or text, and a format.")
    src = args[0]
    rest = list(args[1:])
    fmt = "%f"
    if rest and is_text(rest[0]) and (len(rest) % 2 == 1 or "%" in text(rest[0])):
        fmt = text(rest.pop(0))
    repeat = None
    if rest and not is_text(rest[0]):
        r = float(np.asarray(rest.pop(0), dtype=float).flat[0])
        if r < 0 or (r != int(r) and not np.isinf(r)):
            raise MatlabError("textscan: the number of repetitions must be a non-negative integer or Inf.")
        repeat = None if np.isinf(r) else int(r)
    nv = name_values(rest, _TS_KNOWN, "textscan")
    for key in ("DateLocale", "ExpChars", "BufSize"):
        nv.pop(key, None)  # locale and buffer hints have no effect on the result here
    fh = None
    start = 0
    if is_text(src):
        content = text(src)
    else:
        fh = file_handle(interp, to_int(src, "file identifier"))
        start = fh.tell()
        content = fh.read()
    sc = _Scanner(content, fmt, nv)
    if "HeaderLines" in nv:
        count = to_int(nv["HeaderLines"], "HeaderLines")
        if count < 0:
            raise MatlabError("textscan: HeaderLines must not be negative.")
        sc.skip_header(count)
    sc.run(repeat)
    as_string = "TextType" in nv and _choice(nv, "TextType", ("char", "string"), "char", "textscan") == "string"
    out = [_ts_column(spec, col, as_string) for spec, col in zip(sc.specs, sc.columns) if not spec["skip"]]
    if "CollectOutput" in nv and _flag(nv["CollectOutput"]):
        merged: list = []
        for v in out:
            prev = merged[-1] if merged else None
            if isinstance(v, np.ndarray) and isinstance(prev, np.ndarray) and v.dtype == prev.dtype \
                    and v.shape[0] == prev.shape[0] and v.dtype.kind != "U":
                merged[-1] = np.hstack([prev, v])
            elif isinstance(v, Cell) and isinstance(prev, Cell) and v.data.shape[0] == prev.data.shape[0]:
                merged[-1] = Cell(np.hstack([prev.data, v.data]))
            elif isinstance(v, StringArray) and isinstance(prev, StringArray) \
                    and v.data.shape[0] == prev.data.shape[0]:
                merged[-1] = StringArray(np.hstack([prev.data, v.data]))
            else:
                merged.append(v)
        out = merged
    result = Cell.from_list(out)
    position = float(sc.p)
    if fh is not None:
        consumed = len(fh.encode(content[:sc.p]))
        fh.seek(start + consumed)
        if sc.p >= len(content):
            fh.hit_eof = True
        position = float(start + consumed)
    return (result, position) if nargout > 1 else result
