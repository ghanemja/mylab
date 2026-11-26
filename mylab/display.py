"""Formatting values for the command window (``x = ...`` output and ``disp``).

The layout follows what the MATLAB command window prints: a blank line, ``x =``, a blank
line, the value and a closing blank line with ``format loose`` (the default), and the same
without the blank lines with ``format compact``. Numbers are laid out in columns whose
width depends on the output format (``format short``, ``long``, ``shortG`` ...); a matrix
that is wider than the window is split into ``Columns 1 through N`` blocks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import ast as A
from ._lazy import lazy_function
from .values import Cell, FunctionHandle, MString, StringArray, Struct, class_name, is_char, to_str

Fraction = lazy_function("fractions", "Fraction")  # for format rat; imported on first use

TIMES = "×"  # multiplication sign used in size strings, e.g. 2x3

_MODES = {"short": "short", "long": "long", "shortg": "shortG", "longg": "longG", "shorte": "shortE",
          "longe": "longE", "bank": "bank", "hex": "hex", "rat": "rational", "rational": "rational",
          "shorteng": "shortEng", "longeng": "longEng", "+": "+", "plus": "+"}


class DisplayFormat:
    """Current output format (set by the ``format`` command)."""

    mode = "short"      # short, long, shortg, longg, shorte, longe, bank, hex, rat, shorteng, longeng, +
    spacing = "loose"   # loose or compact
    width = 80          # width of the command window in characters

    def reset(self) -> None:
        self.mode = "short"
        self.spacing = "loose"

    @property
    def long(self) -> bool:
        return self.mode.startswith("long")

    @long.setter
    def long(self, value: bool) -> None:
        self.mode = "long" if value else "short"

    @property
    def style(self) -> str:  # "", "g" or "e"
        return {"shortg": "g", "longg": "g", "shorte": "e", "longe": "e"}.get(self.mode, "")

    @property
    def compact(self) -> bool:
        return self.spacing == "compact"

    @property
    def numeric_name(self) -> str:
        """The name MATLAB reports for the numeric format, e.g. 'shortG'."""
        return _MODES.get(self.mode, self.mode)

    def set_mode(self, name: str) -> bool:
        key = name.lower().replace(" ", "")
        if key in ("rational",):
            key = "rat"
        if key == "plus":
            key = "+"
        if key in _MODES:
            self.mode = key
            return True
        return False


FORMAT = DisplayFormat()


# -- scalars --------------------------------------------------------------------

def _special(x: float) -> str | None:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return None


def _digits(single: bool = False) -> int:
    """Digits after the decimal point for the fixed and exponent layouts of the current format."""
    if FORMAT.long:
        return 7 if single else 15
    return 4


def _exp_text(x: float, digits: int) -> str:
    return f"{x:.{digits}e}"


def _eng_text(x: float, long: bool) -> str:
    """Engineering notation: an exponent that is a multiple of three, three exponent digits."""
    if x == 0:
        return "0.0000e+000" if not long else "0.00000000000000e+000"
    e = int(math.floor(math.log10(abs(x))))
    e3 = 3 * (e // 3)
    mant = x / 10.0 ** e3
    sig = 15 if long else 5
    dec = max(sig - 1 - (e - e3), 0) if long else 4
    text = f"{mant:.{dec}f}"
    if abs(float(text)) >= 1000:
        e3 += 3
        mant = x / 10.0 ** e3
        text = f"{mant:.{dec}f}"
    return f"{text}e{'+' if e3 >= 0 else '-'}{abs(e3):03d}"


def _g_text(x: float, long: bool, single: bool = False) -> str:
    sig = (7 if single else 15) if long else 5
    text = f"{x:.{sig}g}"
    if "e" in text:
        mant, exp = text.split("e")
        sign = exp[0] if exp[0] in "+-" else "+"
        digits = exp.lstrip("+-")
        return f"{mant}e{sign}{digits.zfill(2)}"
    return text


def _hex_text(x: Any) -> str:
    a = np.asarray(x)
    if a.dtype.kind in "iu":
        return format(int(a) & ((1 << (8 * a.dtype.itemsize)) - 1), f"0{2 * a.dtype.itemsize}x")
    if a.dtype == np.float32:
        return format(int(a.view(np.uint32)), "08x")
    return format(int(np.float64(a).view(np.uint64)), "016x")


def rat_text(x: float, length: int = 13) -> str:
    """A rational approximation ``p/q`` the way ``rats`` and ``format rat`` write it."""
    sp = _special(x)
    if sp is not None:
        return sp
    if x == 0:
        return "0"
    if x == int(x) and abs(x) < 10.0 ** (length - 1):
        return str(int(x))
    tol = 10.0 ** (-((length - 1) // 2)) * abs(x)
    # continued fraction expansion until the approximation is within the tolerance
    frac = Fraction(int(math.floor(x)))
    rest = x - math.floor(x)
    h0, h1 = 1, int(math.floor(x))
    k0, k1 = 0, 1
    y = x
    for _ in range(40):
        if abs(x - h1 / k1) <= tol:
            break
        rest = y - math.floor(y)
        if rest == 0:
            break
        y = 1.0 / rest
        a = int(math.floor(y))
        h0, h1 = h1, a * h1 + h0
        k0, k1 = k1, a * k1 + k0
    frac = Fraction(h1, k1)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def _is_int_valued(finite: np.ndarray) -> bool:
    return finite.size == 0 or bool(np.all(finite == np.round(finite)))


def _fixed_or_exp(x: float, digits: int, upper: float) -> str:
    """One finite, non-integer number in ``format short`` / ``long`` when shown on its own."""
    ax = abs(x)
    if 1e-3 < ax < upper:
        text = f"{x:.{digits}f}"
        if abs(float(text)) < upper:
            return text
    return _exp_text(x, digits)


def _scalar_text(x: float, single: bool = False, in_matrix: bool = False) -> str:
    """Text of one real number in the current format, without padding."""
    sp = _special(x)
    if sp is not None:
        return sp
    mode = FORMAT.mode
    if mode == "hex":
        return _hex_text(np.float32(x) if single else np.float64(x))
    if mode == "rat":
        return rat_text(x)
    if mode == "+":
        return "+" if x > 0 else ("-" if x < 0 else " ")
    if mode == "bank":
        return f"{x:.2f}"
    if x == 0:
        return "0"
    is_int = x == math.floor(x)
    if mode in ("shortg", "longg"):
        if is_int and abs(x) < 1e15:
            return str(int(x))
        return _g_text(x, mode == "longg", single)
    if is_int and abs(x) < 1e9:
        return str(int(x))
    if mode in ("shorte", "longe"):
        return _exp_text(x, _digits(single))
    if mode in ("shorteng", "longeng"):
        return _eng_text(x, mode == "longeng")
    digits = _digits(single)
    return _fixed_or_exp(x, digits, 100.0 if FORMAT.long else 1000.0)


# column widths of real numbers per format: (integers up to 3 digits, wider integers, fractions)
def _frac_width(single: bool) -> int:
    mode = FORMAT.mode
    if mode == "long":
        return 12 if single else 20
    if mode == "shorte":
        return 13
    if mode == "longe":
        return 16 if single else 26
    if mode == "shortg":
        return 13
    if mode == "longg":
        return 16 if single else 26
    if mode == "bank":
        return 13
    if mode == "hex":
        return 11 if single else 19
    if mode == "rat":
        return 14
    if mode == "shorteng":
        return 16
    if mode == "longeng":
        return 26
    return 10


def _real_cells(a: np.ndarray) -> tuple[list[str], list[list[str]], int]:
    """(header lines, cell texts by row, column width) for a real floating-point matrix."""
    single = a.dtype == np.float32
    real = a.astype(float)
    mode = FORMAT.mode
    finite = real[np.isfinite(real)]
    maxabs = float(np.max(np.abs(finite))) if finite.size else 0.0
    rows, cols = real.shape

    def grid(fn) -> list[list[str]]:
        return [[fn(float(v)) for v in row] for row in real]

    if mode == "+":
        return [], grid(lambda v: "+" if v > 0 else ("-" if v < 0 else " ")), 1
    if mode == "hex":
        return [], grid(lambda v: _hex_text(np.float32(v) if single else np.float64(v))), _frac_width(single)
    if mode == "rat":
        cells = grid(rat_text)
        return [], cells, max(_frac_width(single), max(len(c) for r in cells for c in r) + 3)
    if mode == "bank":
        cells = grid(lambda v: _special(v) or f"{v:.2f}")
        return [], cells, max(13, max(len(c) for r in cells for c in r) + 2)

    all_int = _is_int_valued(finite)
    if all_int and maxabs < 1e9 and mode not in ("shortg", "longg"):
        cells = grid(lambda v: _special(v) or str(int(v)))
        longest = max(len(c) for r in cells for c in r)
        width = 6 if maxabs <= 999 else 12
        return [], cells, max(width, longest + 2)

    if mode in ("shortg", "longg"):
        if all_int and maxabs < 1e15:
            cells = grid(lambda v: _special(v) or str(int(v)))
            longest = max(len(c) for r in cells for c in r)
            width = 6 if maxabs <= 999 else 12
            return [], cells, max(width, longest + 2)
        cells = grid(lambda v: _special(v) or ("0" if v == 0 else _g_text(v, mode == "longg", single)))
        width = _frac_width(single)
        return [], cells, max(width, max(len(c) for r in cells for c in r) + 2)

    digits = _digits(single)
    width = _frac_width(single)
    if mode in ("shorte", "longe"):
        cells = grid(lambda v: _special(v) or ("0" if v == 0 else _exp_text(v, digits)))
        return [], cells, max(width, max(len(c) for r in cells for c in r) + 2)
    if mode in ("shorteng", "longeng"):
        cells = grid(lambda v: _special(v) or ("0" if v == 0 else _eng_text(v, mode == "longeng")))
        return [], cells, max(width, max(len(c) for r in cells for c in r) + 2)

    # format short / long
    upper = 100.0 if FORMAT.long else 1000.0
    if a.size == 1:
        text = _scalar_text(float(real.flat[0]), single)
        w = width if "e" not in text else (13 if digits == 4 else (16 if single else 26))
        return [], [[text]], max(w, len(text) + 2)
    header: list[str] = []
    scale = 1.0
    if maxabs > 0 and (maxabs >= upper or maxabs < 1e-3):
        exp = int(math.floor(math.log10(maxabs)))
        if maxabs < 1e-3:
            exp += 1
        elif f"{maxabs / 10.0 ** exp:.{digits}f}".startswith("10"):
            exp += 1
        scale = 10.0 ** exp
        header = [f"   1.0e{'+' if exp >= 0 else '-'}{abs(exp):02d} *", ""]

    def fmt(v: float) -> str:
        sp = _special(v)
        if sp is not None:
            return sp
        if v == 0:
            return "0"
        return f"{v / scale:.{digits}f}"

    cells = grid(fmt)
    return header, cells, max(width, max(len(c) for r in cells for c in r) + 1)


def _complex_parts(values: list[complex], long: bool | None = None, style: str | None = None,
                   single: bool = False, scalar: bool = True) -> tuple[list[tuple[str, str, str]], float]:
    """Format complex numbers consistently: [(real text, sign, imaginary magnitude text)], scale."""
    mode = FORMAT.mode
    parts = [(float(np.real(z)), float(np.imag(z))) for z in values]
    finite = [abs(p) for re, im in parts for p in (re, im) if math.isfinite(p)]
    big = max(finite, default=0.0)
    digits = _digits(single)
    upper = 100.0 if FORMAT.long else 1000.0
    scale = 1.0
    use_exp = False
    if mode in ("short", "long") and big > 0 and (big >= upper or big < 1e-3):
        if scalar:
            use_exp = True
        else:
            exp = int(math.floor(math.log10(big)))
            if big < 1e-3:
                exp += 1
            scale = 10.0 ** exp

    def fmt(x: float) -> str:
        sp = _special(x)
        if sp is not None:
            return sp
        if mode in ("shortg", "longg"):
            return "0" if x == 0 else _g_text(x, mode == "longg", single)
        if mode in ("shorte", "longe") or use_exp:
            return _exp_text(x, digits)
        if mode in ("shorteng", "longeng"):
            return _eng_text(x, mode == "longeng")
        if mode == "bank":
            return f"{x:.2f}"
        if mode == "hex":
            return _hex_text(np.float32(x) if single else np.float64(x))
        if mode == "rat":
            return rat_text(x)
        return f"{x / scale:.{digits}f}"

    out = []
    for re, im in parts:
        sign = "-" if im < 0 or (im == 0 and math.copysign(1, im) < 0) else "+"
        out.append((fmt(re), sign, fmt(abs(im))))
    return out, scale


def format_scalar(x: Any, long: bool | None = None) -> str:
    """One number as text in the current format, without the leading padding of a display."""
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    saved = FORMAT.mode
    if long is not None and long != FORMAT.long and FORMAT.mode in ("short", "long"):
        FORMAT.mode = "long" if long else "short"
    try:
        if isinstance(x, (complex, np.complexfloating)):
            single = isinstance(x, np.complex64)
            parts, _ = _complex_parts([complex(x)], single=single)
            re, sign, im = parts[0]
            return f"{re} {sign} {im}i"
        if isinstance(x, (np.integer, int)):
            if FORMAT.mode == "hex" and isinstance(x, np.integer):
                return _hex_text(x)
            return str(int(x))
        return _scalar_text(float(x), isinstance(x, np.float32))
    finally:
        FORMAT.mode = saved


def format_scalar_plain(x: Any) -> str:
    """Compact formatting used when converting numbers to text (e.g. ``"a" + 1``)."""
    if isinstance(x, (complex, np.complexfloating)):
        saved = FORMAT.mode
        FORMAT.mode = "short"
        try:
            z = complex(x)
            parts = []
            for p in (z.real, abs(z.imag)):
                parts.append(_special(p) or (str(int(p)) if p == int(p) and abs(p) < 1e15 else f"{p:.5g}"))
            sign = "-" if z.imag < 0 else "+"
            return f"{parts[0]} {sign} {parts[1]}i"
        finally:
            FORMAT.mode = saved
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, np.integer):
        return str(int(x))
    x = float(x)
    sp = _special(x)
    if sp is not None:
        return sp
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.5g}" if abs(x) >= 1e-4 else f"{x:g}"


# -- matrices -------------------------------------------------------------------

def _wrap_columns(header: list[str], cells: list[list[str]], widths: list[int], gap: str = "",
                  lead: str = "", left: bool = False) -> list[str]:
    """Lay out a grid of cell texts, splitting it into ``Columns a through b`` blocks when it is
    wider than the command window."""
    ncols = len(widths)
    if ncols == 0:
        return list(header)
    limit = max(FORMAT.width - 1, 20)
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < ncols:
        used = len(lead) + widths[start]
        stop = start + 1
        while stop < ncols and used + len(gap) + widths[stop] <= limit:
            used += len(gap) + widths[stop]
            stop += 1
        blocks.append((start, stop))
        start = stop

    def row_text(row: list[str], a: int, b: int) -> str:
        parts = [(c.ljust(widths[j]) if left else c.rjust(widths[j])) for j, c in zip(range(a, b), row[a:b])]
        text = lead + gap.join(parts)
        return text.rstrip() if left else text

    if len(blocks) == 1:
        return header + [row_text(r, 0, ncols) for r in cells]
    out = list(header)
    for k, (a, b) in enumerate(blocks):
        if k:
            out.append("")
        out.append(f"  Columns {a + 1} through {b}" if b - a > 1 else f"  Column {a + 1}")
        out.append("")
        out += [row_text(r, a, b) for r in cells]
    return out


def _numeric_grid(a: np.ndarray) -> tuple[list[str], list[list[str]], int]:
    """(header, cell texts, column width) of a 2-D numeric or logical matrix."""
    kind = a.dtype.kind
    if kind == "b":
        return [], [[str(int(v)) for v in row] for row in a], 4
    if kind in "iu":
        if FORMAT.mode == "hex":
            cells = [[_hex_text(v) for v in row] for row in a]
        else:
            cells = [[str(int(v)) for v in row] for row in a]
        return [], cells, max(len(c) for r in cells for c in r) + 3
    if kind == "c":
        single = a.dtype == np.complex64
        flat, scale = _complex_parts([complex(v) for v in a.flat], single=single, scalar=a.size == 1)
        wr = max(len(p[0]) for p in flat)
        wi = max(len(p[2]) for p in flat)
        digits = _digits(single)
        sample = next((p[2] for p in flat if p[2][:1].isdigit()), "")
        if "e" in sample:  # exponent notation: one space after the sign
            wi, field = wi + 1, (13 if digits == 4 else 26)
        elif "." in sample and FORMAT.mode in ("short", "long"):
            wi, field = max(wi, digits + 3), (9 if digits == 4 else digits + 5)
        else:
            wi, field = wi + 1, wr + 3
        texts = [f"{re:>{wr}} {sign}{im:>{wi}}i" for re, sign, im in flat]
        cells = [texts[r * a.shape[1]:(r + 1) * a.shape[1]] for r in range(a.shape[0])]
        header = []
        if scale != 1.0:
            exp = int(round(math.log10(scale)))
            header = [f"   1.0e{'+' if exp >= 0 else '-'}{abs(exp):02d} *", ""]
        return header, cells, max(field, wr + 1) - wr + len(texts[0])
    return _real_cells(a)


def _matrix_lines(a: np.ndarray) -> list[str]:
    """Body lines for a 2-D numeric / logical matrix (no name header)."""
    header, cells, width = _numeric_grid(a)
    return _wrap_columns(header, cells, [width] * a.shape[1])


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
    return ["".join(c if c else "\0" for c in row) for row in a]


def _shape_word(shape: tuple) -> str:
    if len(shape) > 2:
        return "array"
    if shape[0] == 1 and shape[1] != 1:
        return "row vector"
    if shape[1] == 1 and shape[0] != 1:
        return "column vector"
    return "matrix"


# -- short summaries (inside cells / structs) -------------------------------------

def _row_summary(v: np.ndarray) -> str | None:
    """'1 2 3' for a short numeric row vector, None when it is too long to show inline."""
    if v.dtype.kind == "c":
        parts = [format_scalar(x) for x in v.flat]
    elif v.dtype.kind == "f":
        finite = v[np.isfinite(v)]
        if _is_int_valued(finite) or FORMAT.mode not in ("short", "long"):
            parts = [format_scalar(x) for x in v.flat]
        else:
            digits = _digits(v.dtype == np.float32)
            big = float(np.max(np.abs(finite))) if finite.size else 0.0
            if big >= (100.0 if FORMAT.long else 1000.0) or 0 < big < 1e-3:
                parts = [format_scalar(x) for x in v.flat]
            else:
                parts = [_special(float(x)) or ("0" if x == 0 else f"{float(x):.{digits}f}") for x in v.flat]
    else:
        parts = [format_scalar(x) for x in v.flat]
    text = " ".join(parts)
    return text if len(text) <= 60 and v.size <= 30 else None


def summary(v: Any, in_cell: bool) -> str:
    """How a value is shown inside a cell array (``in_cell``) or as a struct field."""
    if hasattr(v, "summary_text"):
        return v.summary_text(in_cell)
    if isinstance(v, MString):
        return f'["{v}"]' if in_cell else f'"{v}"'
    if isinstance(v, StringArray):
        if v.data.size == 1:
            return "[<missing>]" if in_cell else "<missing>"
        return f"{size_str(v.shape)} string" if in_cell else f"[{size_str(v.shape)} string]"
    if isinstance(v, FunctionHandle):
        return function_handle_text(v)
    if isinstance(v, Cell):
        if v.data.size == 0 and not in_cell:
            return "{}" if v.shape == (0, 0) else f"{{{size_str(v.shape)} cell}}"
        return f"{size_str(v.shape)} cell" if in_cell else f"{{{size_str(v.shape)} cell}}"
    if isinstance(v, Struct):
        return f"{size_str(v.shape)} struct" if in_cell else f"[{size_str(v.shape)} struct]"
    if is_char(v):
        if v.size == 0:
            if in_cell:
                return f"{size_str(v.shape)} char"
            return "''" if v.shape == (0, 0) else f"[{size_str(v.shape)} char]"
        if v.ndim == 2 and v.shape[0] == 1:
            return f"'{to_str(v)}'"
        return f"{size_str(v.shape)} char" if in_cell else f"[{size_str(v.shape)} char]"
    if isinstance(v, np.ndarray):
        if v.size == 0:
            if v.shape == (0, 0) and not in_cell:
                return "[]"
            return f"{size_str(v.shape)} {class_name(v)}" if in_cell else f"[{size_str(v.shape)} {class_name(v)}]"
        if v.size == 1:
            s = format_scalar(v.flat[0])
            return f"[{s}]" if in_cell else s
        if v.ndim == 2 and v.shape[0] == 1:
            parts = _row_summary(v)
            if parts is not None:
                return f"[{parts}]"
        return f"{size_str(v.shape)} {class_name(v)}" if in_cell else f"[{size_str(v.shape)} {class_name(v)}]"
    if hasattr(v, "matlab_shape") and hasattr(v, "class_name"):
        text = f"{size_str(v.matlab_shape())} {v.class_name()}"
        return text if in_cell else f"[{text}]"
    return str(v)


def _cell_lines(c: Cell) -> list[str]:
    if c.data.size == 0:
        return []
    texts = [[summary(x, True) for x in row] for row in c.data]
    widths = [max(len(texts[r][j]) for r in range(len(texts))) for j in range(c.shape[1])]

    def pad(t: str, w: int) -> str:
        # values in brackets are right-aligned inside them, everything else is left-aligned
        if t.startswith("[") and t.endswith("]"):
            return "{[" + " " * (w - len(t)) + t[1:] + "}"
        return "{" + t + " " * (w - len(t)) + "}"

    cells = [[pad(t, widths[j]) for j, t in enumerate(row)] for row in texts]
    return _wrap_columns([], cells, [w + 2 for w in widths], gap="    ", lead="    ", left=True)


def _string_cells(v: StringArray) -> list[str]:
    data = v.data
    cells = [["<missing>" if x is None else f'"{x}"' for x in row] for row in data]
    widths = [max(len(cells[r][j]) for r in range(len(cells))) for j in range(data.shape[1])]
    return _wrap_columns([], cells, widths, gap="    ", lead="    ", left=True)


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

def _compact_text(text: str) -> str:
    """Drop the blank lines of a loose layout (``format compact``)."""
    lines = [ln for ln in text.split("\n") if ln.strip() != ""]
    return "\n".join(lines) + "\n" if lines else ""


def finish_layout(text: str) -> str:
    """Apply the line spacing setting to a text that was laid out for ``format loose``."""
    return _compact_text(text) if FORMAT.compact else text


def numeric_header(v: np.ndarray) -> str | None:
    """The class line printed above logical, integer and single values, e.g. '  2×2 int8 matrix'."""
    if v.dtype == bool:
        return "  logical" if v.size == 1 else f"  {size_str(v.shape)} logical array"
    if v.dtype.kind in "iu" or v.dtype in (np.float32, np.complex64):
        cls = class_name(v)
        if v.size == 1:
            return f"  {cls}"
        return f"  {size_str(v.shape)} {cls} {_shape_word(v.shape)}"
    return None


def empty_text(v: np.ndarray) -> str:
    """'  0×3 empty double matrix' and friends."""
    cls = class_name(v)
    word = "array" if cls in ("char", "logical") else _shape_word(v.shape)
    return f"  {size_str(v.shape)} empty {cls} {word}"


def display_lines(v: Any) -> list[str]:
    """Lines printed by ``disp(v)`` (laid out for ``format loose``)."""
    if isinstance(v, np.ndarray) and v.ndim > 2 and v.size:
        lines: list[str] = []
        for suffix, page in _pages(v):
            lines += ["", suffix, ""]
            lines += _char_rows(page) if is_char(page) else _matrix_lines(page)
            lines.append("")
        return lines
    if isinstance(v, (Cell, StringArray)) and v.data.ndim > 2 and v.data.size:
        lines = []
        for suffix, page in _pages(v.data):
            lines += ["", suffix, ""]
            lines += _cell_lines(Cell(page)) if isinstance(v, Cell) else _string_cells(StringArray(page))
            lines.append("")
        return lines
    if isinstance(v, StringArray):
        if v.data.size == 0:
            return []
        if v.data.size == 1:
            return ["<missing>"]
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
        if v.data.size == 1:
            return _struct_lines(v)
        head = f"  {size_str(v.data.shape)} {'empty ' if v.data.size == 0 else ''}struct array with "
        if not v.fields:
            return [head + "no fields."]
        return [head + "fields:", ""] + ["    " + f for f in v.fields]
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return []
        return _matrix_lines(v)
    if hasattr(v, "disp_lines"):
        return v.disp_lines()
    if hasattr(v, "display_lines"):
        return v.display_lines()
    return [str(v)]


def disp_text(v: Any) -> str:
    """Everything ``disp(v)`` prints, with the current line spacing applied."""
    lines = display_lines(v)
    if not lines:
        return ""
    return finish_layout("\n".join(lines) + "\n")


def _paged(name: str, data: np.ndarray, eq: str, body) -> str:
    out = []
    for suffix, page in _pages(data):
        out.append(f"\n{name}{suffix} {eq}\n\n" + "\n".join(body(page)) + "\n\n")
    return "".join(out)


def _loose_display(name: str, v: Any) -> str:
    if hasattr(v, "format_display"):
        text = v.format_display(name)
        if getattr(v, "is_sym", False) or text.startswith("\n") or not text:
            return text
        return "\n" + text
    if isinstance(v, np.ndarray):
        if v.size == 0:
            if v.shape == (0, 0) and v.dtype.kind == "f" and v.dtype != np.float32:
                return f"\n{name} =\n\n     []\n\n"
            return f"\n{name} =\n\n{empty_text(v)}\n\n"
        if v.ndim > 2:
            if is_char(v):
                head = f"\n{name} =\n\n  {size_str(v.shape)} char array\n"
                return head + _paged(name, v, "=", lambda p: [f"    '{r}'" for r in _char_rows(p)])
            typ = numeric_header(v)
            head = f"\n{name} =\n\n{typ}\n" if typ else ""
            return head + _paged(name, v, "=", _matrix_lines)
        if is_char(v):
            if v.shape[0] == 1:
                return f"\n{name} =\n\n    '{_char_rows(v)[0]}'\n\n"
            rows = "\n".join(f"    '{r}'" for r in _char_rows(v))
            return f"\n{name} =\n\n  {size_str(v.shape)} char array\n\n{rows}\n\n"
        typ = numeric_header(v)
        head = f"\n{name} =\n\n" + (f"{typ}\n\n" if typ else "")
        return head + "\n".join(_matrix_lines(v)) + "\n\n"
    if isinstance(v, MString):
        return f'\n{name} = \n\n    "{v}"\n\n'
    if isinstance(v, StringArray):
        if v.data.size == 0:
            return f"\n{name} = \n\n  {size_str(v.shape)} empty string array\n\n"
        if v.data.size == 1:
            return f"\n{name} = \n\n    <missing>\n\n"
        head = f"\n{name} = \n\n  {size_str(v.shape)} string array\n"
        if v.data.ndim > 2:
            return head + _paged(name, v.data, "= ", lambda p: _string_cells(StringArray(p)))
        return head + "\n" + "\n".join(_string_cells(v)) + "\n\n"
    if isinstance(v, FunctionHandle):
        return f"\n{name} =\n\n  function_handle with value:\n\n    {function_handle_text(v)}\n\n"
    if isinstance(v, Cell):
        if v.data.size == 0:
            return f"\n{name} =\n\n  {size_str(v.shape)} empty cell array\n\n"
        head = f"\n{name} =\n\n  {size_str(v.shape)} cell array\n"
        if v.data.ndim > 2:
            return head + _paged(name, v.data, "=", lambda p: _cell_lines(Cell(p)))
        return head + "\n" + "\n".join(_cell_lines(v)) + "\n\n"
    if isinstance(v, Struct):
        if v.data.size == 1:
            body = "\n".join(_struct_lines(v))
            return f"\n{name} = \n\n  struct with fields:\n\n{body}\n\n" if body else \
                f"\n{name} = \n\n  struct with no fields.\n\n"
        return f"\n{name} = \n\n" + "\n".join(display_lines(v)) + "\n\n"
    return f"\n{name} =\n\n    {v!r}\n\n"


def format_display(name: str, v: Any) -> str:
    """Text printed for an unsuppressed statement such as ``x = 5``."""
    return finish_layout(_loose_display(name, v))


def object_header(name: str) -> str:
    """The ``name = `` lines printed before an object that displays itself with ``disp``."""
    return finish_layout(f"\n{name} = \n\n")
