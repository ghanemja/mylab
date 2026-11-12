"""Formatting values for the command window (``x = ...`` output and ``disp``)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import ast as A
from .values import Cell, FunctionHandle, MString, StringArray, Struct, class_name, is_char, to_str


def _string_cells(v: StringArray) -> list[str]:
    data = v.data
    cells = [['<missing>' if x is None else f'"{x}"' for x in row] for row in data]
    widths = [max(len(cells[r][j]) for r in range(len(cells))) for j in range(data.shape[1])]
    return ["    " + "    ".join(cells[r][j].ljust(widths[j]) for j in range(len(widths))).rstrip()
            for r in range(len(cells))]

TIMES = "×"  # multiplication sign used in size strings, e.g. 2x3


class DisplayFormat:
    """Current output format (set by the ``format`` command)."""

    mode = "short"  # short, long, shortg, longg, shorte, longe

    @property
    def long(self) -> bool:
        return self.mode.startswith("long")

    @long.setter
    def long(self, value: bool) -> None:
        self.mode = "long" if value else "short"

    @property
    def style(self) -> str:  # "", "g" or "e"
        return {"shortg": "g", "longg": "g", "shorte": "e", "longe": "e"}.get(self.mode, "")


FORMAT = DisplayFormat()


# -- scalars --------------------------------------------------------------------

def _special(x: float) -> str | None:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return None


def _fmt_real(x: float, long: bool, style: str = "") -> str:
    sp = _special(x)
    if sp is not None:
        return sp
    if style == "g":
        return f"{x:.{15 if long else 5}g}"
    if style == "e":
        return f"{x:.{15 if long else 4}e}"
    if x == int(x) and abs(x) < 1e9:
        return str(int(x)) if x != 0 else "0"
    ax = abs(x)
    if long:
        return f"{x:.15f}" if 1e-3 <= ax < 100 else f"{x:.15e}"
    return f"{x:.4f}" if 1e-3 < ax < 1e3 else f"{x:.4e}"


def _complex_parts(values: list[complex], long: bool, style: str = "") -> list[tuple[str, str, str]]:
    """Format complex numbers consistently: (real text, sign, imaginary magnitude text)."""
    parts = [(float(np.real(z)), float(np.imag(z))) for z in values]
    finite = [abs(p) for re, im in parts for p in (re, im) if math.isfinite(p)]
    all_int = all(p == int(p) for re, im in parts for p in (re, im) if math.isfinite(p))
    big = max(finite, default=0.0)
    digits = 15 if long else 4

    def fmt(x: float) -> str:
        sp = _special(x)
        if sp is not None:
            return sp
        if style == "g":
            return f"{x:.{15 if long else 5}g}"
        if style == "e" or (not all_int and (big >= 1e5 or (0 < big < 1e-3))):
            return f"{x:.{digits}e}"
        if all_int and big < 1e9:
            return str(int(x))
        return f"{x:.{digits}f}"

    out = []
    for re, im in parts:
        sign = "-" if im < 0 or (im == 0 and math.copysign(1, im) < 0) else "+"
        out.append((fmt(re), sign, fmt(abs(im))))
    return out


def format_scalar(x: Any, long: bool | None = None) -> str:
    long = FORMAT.long if long is None else long
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (complex, np.complexfloating)):
        re, sign, im = _complex_parts([complex(x)], long, FORMAT.style)[0]
        return f"{re} {sign} {im}i"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    return _fmt_real(float(x), long, FORMAT.style)


def format_scalar_plain(x: Any) -> str:
    """Compact formatting used when converting numbers to text (e.g. ``"a" + 1``)."""
    if isinstance(x, (complex, np.complexfloating)):
        return format_scalar(x)
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.5g}" if abs(x) >= 1e-4 else f"{x:g}"


# -- matrices -------------------------------------------------------------------

def _matrix_lines(a: np.ndarray) -> list[str]:
    """Body lines for a numeric / logical matrix (no name header)."""
    long = FORMAT.long
    style = FORMAT.style
    if a.dtype == bool:
        return ["".join(f"{int(v):>4d}" for v in row) for row in a]
    if a.dtype.kind in "iu":
        width = max(len(str(v)) for v in a.flat) + 2
        width = max(width, 4)
        return ["".join(f"{int(v):>{width}d}" for v in row) for row in a]
    if a.dtype.kind == "c":
        flat = _complex_parts([complex(v) for v in a.flat], long, style)
        wr = max(len(p[0]) for p in flat)
        wi = max(len(p[2]) for p in flat)
        cells = [f"{re:>{wr}} {sign} {im:<{wi}}i" for re, sign, im in flat]
        grid = np.array(cells, dtype=object).reshape(a.shape)
        return ["".join(f"{c:>{wr + wi + 7}}" for c in row) for row in grid]

    real = a.astype(float)
    if style:
        cells = [[_fmt_real(float(v), long, style) for v in row] for row in real]
        width = max(len(c) for r in cells for c in r) + 3
        return ["".join(f"{c:>{width}}" for c in row) for row in cells]
    finite = real[np.isfinite(real)]
    all_int = finite.size == 0 or bool(np.all(finite == np.round(finite)))
    if all_int and (finite.size == 0 or np.max(np.abs(finite)) < 1e9):
        cells = [[_fmt_real(float(v), long) for v in row] for row in real]
        width = max(max(len(c) for r in cells for c in r) + 3, 6)
        return ["".join(f"{c:>{width}}" for c in row) for row in cells]

    digits = 15 if long else 4
    maxabs = float(np.max(np.abs(finite))) if finite.size else 0.0
    header = []
    scale = 1.0
    if a.size > 1 and maxabs > 0 and (maxabs >= (100 if long else 1e3) or maxabs < 1e-3):
        exp = int(math.floor(math.log10(maxabs)))
        scale = 10.0 ** exp
        header = [f"   1.0e{exp:+03d} *", ""]
    elif a.size == 1:
        return ["    " + _fmt_real(float(real.flat[0]), long)]

    def fmt(v: float) -> str:
        sp = _special(v)
        if sp is not None:
            return sp
        if v == 0:
            return "0"
        return f"{v / scale:.{digits}f}"

    cells = [[fmt(float(v)) for v in row] for row in real]
    width = max(10 if not long else 20, max(len(c) for r in cells for c in r) + 3)
    return header + ["".join(f"{c:>{width}}" for c in row) for row in cells]


def size_str(shape: tuple) -> str:
    return TIMES.join(str(d) for d in shape)


def _pages(a: np.ndarray):
    """Yield (suffix, 2-D page) for an N-D array, e.g. ('(:,:,2)', page)."""
    trailing = a.shape[2:]
    for idx in np.ndindex(*trailing[::-1]):
        idx = idx[::-1]  # column-major page order: first trailing index varies fastest
        page = a[(slice(None), slice(None)) + idx]
        yield "(:,:," + ",".join(str(i + 1) for i in idx) + ")", page


def _char_rows(a: np.ndarray) -> list[str]:
    return ["".join(row) for row in a]


# -- short summaries (inside cells / structs) -------------------------------------

def summary(v: Any, in_cell: bool) -> str:
    if hasattr(v, "summary_text"):
        return v.summary_text(in_cell)
    if isinstance(v, MString):
        return f'"{v}"'
    if isinstance(v, StringArray):
        return f"{size_str(v.shape)} string" if in_cell else f"[{size_str(v.shape)} string]"
    if isinstance(v, FunctionHandle):
        return function_handle_text(v)
    if isinstance(v, Cell):
        return f"{size_str(v.shape)} cell" if in_cell else f"{{{size_str(v.shape)} cell}}"
    if isinstance(v, Struct):
        return f"{size_str(v.shape)} struct" if in_cell else f"[{size_str(v.shape)} struct]"
    if is_char(v):
        if v.size == 0 and in_cell:
            return f"{size_str(v.shape)} char"
        if v.shape[0] <= 1:
            return f"'{to_str(v)}'"
        return f"{size_str(v.shape)} char"
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return "[]" if v.shape == (0, 0) else f"{size_str(v.shape)} {class_name(v)}"
        if v.size == 1:
            s = format_scalar(v.flat[0])
            return f"[{s}]" if in_cell else s
        if v.shape[0] == 1 and v.size <= 10:
            parts = " ".join(format_scalar(x) for x in v.flat)
            return f"[{parts}]"
        return f"{size_str(v.shape)} {class_name(v)}" if in_cell else f"[{size_str(v.shape)} {class_name(v)}]"
    return str(v)


def _cell_lines(c: Cell) -> list[str]:
    if c.data.size == 0:
        return []
    cells = [["{" + summary(x, True) + "}" for x in row] for row in c.data]
    widths = [max(len(cells[r][j]) for r in range(len(cells))) for j in range(c.shape[1])]
    return ["    " + "    ".join(cells[r][j].ljust(widths[j]) for j in range(len(widths))).rstrip()
            for r in range(len(cells))]


def _struct_lines(s: Struct) -> list[str]:
    if s.data.size != 1:
        return ["    " + f for f in s.fields]
    elem = s.data.flat[0]
    if not s.fields:
        return []
    w = max(len(f) for f in s.fields)
    return [f"    {f:>{w}}: {summary(elem[f], False)}" for f in s.fields]


# -- unparser for anonymous functions ---------------------------------------------

_PREC = {"||": 1, "&&": 2, "|": 3, "&": 4, "<": 5, "<=": 5, ">": 5, ">=": 5, "==": 5,
         "~=": 5, ":": 6, "+": 7, "-": 7, "*": 8, "/": 8, "\\": 8, ".*": 8, "./": 8,
         ".\\": 8, "^": 10, ".^": 10}


def unparse(node: A.Node, parent_prec: int = 0) -> str:
    if isinstance(node, A.Number):
        v = node.value
        if isinstance(v, complex):
            return f"{format_scalar_plain(v.imag)}i"
        return format_scalar_plain(v) if v == int(v) else repr(float(v))
    if isinstance(node, A.IntLiteral):
        return node.text
    if isinstance(node, A.String):
        q = '"' if node.double else "'"
        return q + node.value.replace(q, q + q) + q
    if isinstance(node, A.Ident):
        return node.name
    if isinstance(node, A.Colon):
        return ":"
    if isinstance(node, A.End):
        return "end"
    if isinstance(node, A.Range):
        parts = [unparse(node.start, 7)]
        if node.step is not None:
            parts.append(unparse(node.step, 7))
        parts.append(unparse(node.stop, 7))
        s = ":".join(parts)
        return f"({s})" if parent_prec > 6 else s
    if isinstance(node, A.BinOp):
        p = _PREC[node.op]
        s = f"{unparse(node.left, p)}{node.op}{unparse(node.right, p + 1)}"
        return f"({s})" if p < parent_prec else s
    if isinstance(node, A.UnaryOp):
        s = node.op + unparse(node.operand, 9)
        return f"({s})" if parent_prec > 9 else s
    if isinstance(node, A.Postfix):
        return unparse(node.operand, 11) + node.op
    if isinstance(node, A.Index):
        return f"{unparse(node.base, 11)}({','.join(unparse(a) for a in node.args)})"
    if isinstance(node, A.CellIndex):
        return f"{unparse(node.base, 11)}{{{','.join(unparse(a) for a in node.args)}}}"
    if isinstance(node, A.Field):
        return f"{unparse(node.base, 11)}.{node.name}"
    if isinstance(node, A.DynField):
        return f"{unparse(node.base, 11)}.({unparse(node.expr)})"
    if isinstance(node, (A.Matrix, A.CellArray)):
        o, c = ("[", "]") if isinstance(node, A.Matrix) else ("{", "}")
        return o + ";".join(",".join(unparse(e) for e in row) for row in node.rows) + c
    if isinstance(node, A.FuncHandle):
        return "@" + node.name
    if isinstance(node, A.Lambda):
        return f"@({','.join(node.params)}){unparse(node.body)}"
    return "?"


def function_handle_text(f: FunctionHandle) -> str:
    if f.is_anonymous:
        return f.text or f"@({','.join(f.params)}){unparse(f.body)}"
    return "@" + str(f.name)


# -- public API -----------------------------------------------------------------

def display_lines(v: Any) -> list[str]:
    """Lines printed by ``disp(v)``."""
    if isinstance(v, np.ndarray) and v.ndim > 2 and v.size:
        lines: list[str] = []
        for suffix, page in _pages(v):
            lines += [f"ans{suffix} =", ""] + display_lines(page) + [""]
        return lines
    if isinstance(v, (Cell, Struct, StringArray)) and v.data.ndim > 2:
        return [f"  {size_str(v.data.shape)} {class_name(v)} array"]
    if isinstance(v, StringArray):
        return _string_cells(v)
    if isinstance(v, MString):
        return [str(v)]
    if is_char(v):
        return _char_rows(v) if v.size else [""]
    if isinstance(v, FunctionHandle):
        return ["    " + function_handle_text(v)]
    if isinstance(v, Cell):
        return _cell_lines(v)
    if isinstance(v, Struct):
        return _struct_lines(v)
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return []
        return _matrix_lines(v)
    if hasattr(v, "display_lines"):
        return v.display_lines()
    return [str(v)]


def format_display(name: str, v: Any) -> str:
    """Text printed for an unsuppressed statement such as ``x = 5``."""
    if hasattr(v, "format_display"):
        return v.format_display(name)
    if isinstance(v, np.ndarray) and v.ndim > 2:
        if v.size == 0:
            return f"{name} =\n\n  {size_str(v.shape)} empty {class_name(v)} array\n\n"
        out = []
        for suffix, page in _pages(v):
            if is_char(page):
                body = "\n".join(f"    '{r}'" for r in _char_rows(page))
            else:
                body = "\n".join(_matrix_lines(page))
            out.append(f"{name}{suffix} =\n\n{body}\n\n")
        return "".join(out)
    if isinstance(v, StringArray):
        if v.data.size == 0:
            return f"{name} = \n\n  {size_str(v.shape)} empty string array\n\n"
        if v.data.ndim > 2:
            return f"{name} =\n\n  {size_str(v.shape)} string array\n\n"
        if v.data.size == 1:
            return f"{name} = <missing>\n"
        return f"{name} = \n\n  {size_str(v.shape)} string array\n\n" + "\n".join(_string_cells(v)) + "\n\n"
    if isinstance(v, (Cell, Struct)) and v.data.ndim > 2:
        kind = "cell array" if isinstance(v, Cell) else "struct array with fields:\n\n" + \
            "\n".join("    " + f for f in v.fields)
        return f"{name} =\n\n  {size_str(v.data.shape)} {kind}\n\n"
    if isinstance(v, MString):
        return f'{name} = "{v}"\n'
    if isinstance(v, FunctionHandle):
        return f"{name} =\n\n  function_handle with value:\n\n    {function_handle_text(v)}\n\n"
    if isinstance(v, Cell):
        if v.data.size == 0:
            return f"{name} =\n\n  {size_str(v.shape)} empty cell array\n\n"
        return f"{name} =\n\n  {size_str(v.shape)} cell array\n\n" + "\n".join(_cell_lines(v)) + "\n\n"
    if isinstance(v, Struct):
        if v.data.size == 1:
            body = "\n".join(_struct_lines(v))
            return f"{name} = \n\n  struct with fields:\n\n{body}\n\n" if body else \
                f"{name} = \n\n  struct with no fields.\n\n"
        return f"{name} = \n\n  {size_str(v.shape)} struct array with fields:\n\n" + \
            "\n".join(_struct_lines(v)) + "\n\n"
    if is_char(v):
        if v.size == 0:
            return f"{name} =\n\n  0{TIMES}0 empty char array\n\n"
        if v.shape[0] == 1:
            return f"{name} = '{to_str(v)}'\n"
        rows = "\n".join(f"    '{r}'" for r in _char_rows(v))
        return f"{name} =\n\n  {size_str(v.shape)} char array\n\n{rows}\n\n"
    if isinstance(v, np.ndarray):
        if v.size == 0:
            if v.shape == (0, 0):
                return f"{name} =\n\n     []\n\n"
            return f"{name} =\n\n  {size_str(v.shape)} empty {class_name(v)} array\n\n"
        typ = ""
        if v.dtype == bool or v.dtype.kind in "iu" or v.dtype in (np.float32, np.complex64):
            cls = class_name(v)
            if v.size == 1:
                typ = f"  {cls}\n\n"
            elif cls == "logical":
                typ = f"  {size_str(v.shape)} logical array\n\n"
            elif v.ndim == 2 and v.shape[0] == 1:
                typ = f"  {size_str(v.shape)} {cls} row vector\n\n"
            elif v.ndim == 2 and v.shape[1] == 1:
                typ = f"  {size_str(v.shape)} {cls} column vector\n\n"
            else:
                typ = f"  {size_str(v.shape)} {cls} matrix\n\n"
        if v.size == 1 and not typ:
            return f"{name} = {format_scalar(v.flat[0])}\n"
        lines = _matrix_lines(v)
        return f"{name} =\n\n{typ}" + "\n".join(lines) + "\n\n"
    return f"{name} = {v!r}\n"
