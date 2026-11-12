"""jsondecode and jsonencode."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, Struct, char, is_char, to_str
from ._registry import builtin
from ._util import is_text, nargs, text


# -- decoding --------------------------------------------------------------------------------------

def _field_name(key: str, used: list[str]) -> str:
    name = re.sub(r"\W", "_", key)
    if not name or not (name[0].isalpha() and name[0].isascii()):
        name = "x" + name
    base, k = name, 1
    while name in used:
        name = f"{base}_{k}"
        k += 1
    return name


def _from_json(v: Any) -> Any:
    if v is None:
        return np.zeros((0, 0))
    if isinstance(v, bool):
        return np.array([[v]])
    if isinstance(v, (int, float)):
        return np.array([[float(v)]])
    if isinstance(v, str):
        return char(v)
    if isinstance(v, dict):
        names: list[str] = []
        values = {}
        for key, item in v.items():
            name = _field_name(key, names)
            names.append(name)
            values[name] = _from_json(item)
        return Struct.scalar(values)
    return _from_list(v)


def _numeric_block(v: list) -> np.ndarray | None:
    """A (nested) list of numbers/nulls or of booleans with a regular shape, as an array; else None."""
    def shape_of(x) -> tuple | None:
        if isinstance(x, list):
            if not x:
                return None
            shapes = [shape_of(item) for item in x]
            if shapes[0] is None or any(s != shapes[0] for s in shapes):
                return None
            return (len(x),) + shapes[0]
        if x is None or (isinstance(x, (int, float)) and not isinstance(x, bool)):
            return ()
        return None

    def all_bool(x) -> bool:
        if isinstance(x, list):
            return bool(x) and all(all_bool(item) for item in x)
        return isinstance(x, bool)

    def bool_shape(x) -> tuple | None:
        if isinstance(x, list):
            shapes = [bool_shape(item) for item in x]
            if not shapes or shapes[0] is None or any(s != shapes[0] for s in shapes):
                return None
            return (len(x),) + shapes[0]
        return ()

    if all_bool(v):
        shape = bool_shape(v)
        return None if shape is None else np.array(v, dtype=bool).reshape(shape)
    shape = shape_of(v)
    if shape is None:
        return None

    def numbers(x):
        if isinstance(x, list):
            return [numbers(item) for item in x]
        return math.nan if x is None else float(x)

    return np.array(numbers(v), dtype=float).reshape(shape)


def _from_list(v: list) -> Any:
    if not v:
        return np.zeros((0, 0))
    block = _numeric_block(v)
    if block is not None:
        return block.reshape(-1, 1) if block.ndim == 1 else block
    items = [_from_json(item) for item in v]
    if all(isinstance(item, Struct) for item in items) and all(
            item.fields == items[0].fields for item in items):
        data = np.empty((len(items), 1), dtype=object)
        for k, item in enumerate(items):
            data[k, 0] = item.data.flat[0]
        return Struct(data, list(items[0].fields))
    return Cell.from_list(items, column=True)


@builtin("jsondecode")
def _jsondecode(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("jsondecode needs JSON text, e.g. jsondecode('{\"a\": [1, 2, 3]}'). "
                          "To read a file use jsondecode(fileread(name)).")
    try:
        data = json.loads(text(args[0]))
    except json.JSONDecodeError as e:
        raise MatlabError(f"JSON syntax error at line {e.lineno}, column {e.colno}: {e.msg}.",
                          "MATLAB:json:ExpectedValue") from None
    return _from_json(data)


# -- encoding --------------------------------------------------------------------------------------

def _number(x: Any, convert: bool) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        if convert:
            return None
        return _Raw("NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity"))
    if x == int(x) and abs(x) < 1e15:
        return int(x)
    return _Raw(f"{x:.15g}")


class _Raw:
    """A number already formatted as JSON text."""

    def __init__(self, s: str):
        self.s = s


def _nested(a: np.ndarray, convert: bool) -> Any:
    if a.ndim == 1:
        return [_number(x, convert) for x in a]
    return [_nested(a[k], convert) for k in range(a.shape[0])]


def _to_json(v: Any, convert: bool) -> Any:
    if isinstance(v, MString):
        return str(v)
    if isinstance(v, StringArray):
        items = [None if x is None else x for x in v.data.flatten(order="F")]
        return items
    if is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return ["".join(row) for row in v]
        return to_str(v)
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            raise MatlabError("jsonencode: complex numbers cannot be written as JSON.")
        if v.size == 0:
            return []
        if v.size == 1:
            return _number(v.flat[0], convert)
        if v.ndim == 2 and 1 in v.shape:
            return [_number(x, convert) for x in v.flatten(order="F")]
        return _nested(v, convert)
    if isinstance(v, Cell):
        return [_to_json(x, convert) for x in v.data.flatten(order="F")]
    if isinstance(v, Struct):
        items = [{f: _to_json(d[f], convert) for f in v.fields} for d in v.data.flatten(order="F")]
        return items[0] if v.data.size == 1 else items
    cls = v.class_name() if hasattr(v, "class_name") else type(v).__name__
    if cls == "containers.Map":
        return {(k if isinstance(k, str) else f"{k:g}"): _to_json(v.store[k], convert) for k in v.sorted_keys()}
    if cls == "table":
        rows = []
        for i in range(v.nrows):
            row = {}
            for name, col in zip(v.names, v.columns):
                if isinstance(col, Cell):
                    row[name] = _to_json(col.data[i, 0], convert)
                elif isinstance(col, StringArray):
                    row[name] = col.data[i, 0]
                elif isinstance(col, np.ndarray):
                    row[name] = _to_json(col[i:i + 1], convert)
                else:
                    row[name] = _to_json(col.index_get(None, [np.array([[i + 1.0]])]), convert)
            rows.append(row)
        return rows
    if hasattr(v, "element_texts"):  # datetime, duration, categorical
        texts = [str(s) for s in np.asarray(v.element_texts()).flatten(order="F")]
        return texts[0] if len(texts) == 1 else texts
    if hasattr(v, "props") and hasattr(v, "cls"):  # a classdef object: its public properties
        return {n: _to_json(v.props.get(n, np.zeros((0, 0))), convert) for n in v.cls.instance_props()}
    raise MatlabError(f"jsonencode: values of class '{cls}' cannot be written as JSON.")


def _dump(obj: Any, pretty: bool, level: int = 0) -> str:
    pad, inner = ("  " * level, "  " * (level + 1)) if pretty else ("", "")
    nl = "\n" if pretty else ""
    if isinstance(obj, _Raw):
        return obj.s
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        sep = ": " if pretty else ":"
        parts = [f"{inner}{json.dumps(k, ensure_ascii=False)}{sep}{_dump(val, pretty, level + 1)}"
                 for k, val in obj.items()]
        return "{" + nl + ("," + nl).join(parts) + nl + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        parts = [f"{inner}{_dump(val, pretty, level + 1)}" for val in obj]
        return "[" + nl + ("," + nl).join(parts) + nl + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


@builtin("jsonencode")
def _jsonencode(interp, args, nargout):
    nargs(args, 1, None, "jsonencode")
    convert, pretty = True, False
    k = 1
    while k < len(args):
        if k + 1 >= len(args):
            raise MatlabError("jsonencode: options come in name-value pairs.")
        key = text(args[k], "option name").lower()
        flag = bool(np.asarray(args[k + 1]).flat[0])
        if key == "convertinfandnan":
            convert = flag
        elif key == "prettyprint":
            pretty = flag
        else:
            raise MatlabError(f"jsonencode: unknown option '{text(args[k])}'. Use 'PrettyPrint' or "
                              "'ConvertInfAndNaN'.")
        k += 2
    return char(_dump(_to_json(args[0], convert), pretty))
