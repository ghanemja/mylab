"""Excel workbooks (.xlsx): xlsread, xlswrite, sheetnames, and Excel support for
readtable, writetable, readmatrix, writematrix, readcell and writecell.

When the ``openpyxl`` package is installed (``pip install mylab[excel]``) it does
the reading and writing. Without it a small built-in reader and writer handles
plain workbooks: numbers, text and logical values, no formatting.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import zipfile
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, char, is_char, to_str
from . import BUILTINS
from ._registry import builtin
from ._util import is_text, text

EXTENSIONS = (".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".xlsb", ".ods")
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "p": "http://schemas.openxmlformats.org/package/2006/relationships"}
_EPOCH = _dt.datetime(1899, 12, 30)


def is_excel_file(name: Any) -> bool:
    return is_text(name) and os.path.splitext(text(name))[1].lower() in EXTENSIONS


def _check_format(fname: str, what: str) -> None:
    ext = os.path.splitext(fname)[1].lower()
    if ext in (".xls", ".xlsb", ".ods"):
        raise MatlabError(f"{what}: '{ext}' spreadsheets are not supported. Save the file as an Excel "
                          "workbook (.xlsx) or as .csv and read that.")


def _openpyxl():
    try:
        import openpyxl
        return openpyxl
    except ImportError:
        return None


# -- cell references -------------------------------------------------------------------------------

def _col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _col_letters(index: int) -> str:
    s = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        s = chr(65 + rem) + s
    return s


def _parse_range(rng: str | None, what: str) -> tuple[int, int, int | None, int | None]:
    """(first row, first column, last row, last column), 0-based; last values None for an open range."""
    if not rng:
        return 0, 0, None, None
    m = re.fullmatch(r"\$?([A-Za-z]+)\$?(\d+)(?::\$?([A-Za-z]+)\$?(\d+))?", rng.strip())
    if not m:
        raise MatlabError(f"{what}: '{rng}' is not a cell range. Write it like 'B2' or 'A1:C10'.")
    r0, c0 = int(m.group(2)) - 1, _col_index(m.group(1))
    if m.group(3):
        return r0, c0, int(m.group(4)) - 1, _col_index(m.group(3))
    return r0, c0, None, None


# -- reading -----------------------------------------------------------------------------------------

def _builtin_sheets(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    targets = {r.get("Id"): r.get("Target") for r in rels.findall("p:Relationship", _NS)}
    sheets = []
    for s in wb.findall("m:sheets/m:sheet", _NS):
        target = targets.get(s.get(f"{{{_NS['r']}}}id"), "")
        path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        sheets.append((s.get("name"), path))
    return sheets


def _builtin_read(fname: str, sheet: Any) -> tuple[list[list], list[str]]:
    try:
        z = zipfile.ZipFile(fname)
    except zipfile.BadZipFile:
        raise MatlabError(f"'{fname}' is not an Excel workbook (.xlsx) that can be read.") from None
    with z:
        sheets = _builtin_sheets(z)
        names = [n for n, _ in sheets]
        path = sheets[_pick_sheet(names, sheet, fname)][1]
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        root = ET.fromstring(z.read(path))
    rows: list[list] = []
    for r_el in root.findall("m:sheetData/m:row", _NS):
        r = int(r_el.get("r", len(rows) + 1)) - 1
        while len(rows) <= r:
            rows.append([])
        row = rows[r]
        next_col = 0
        for c in r_el.findall("m:c", _NS):
            ref = c.get("r")
            col = _col_index(re.match(r"[A-Za-z]+", ref).group(0)) if ref else next_col
            next_col = col + 1
            kind = c.get("t", "n")
            v = c.find("m:v", _NS)
            value: Any = None
            if kind == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
            elif v is not None and v.text is not None:
                if kind == "s":
                    value = shared[int(v.text)]
                elif kind == "b":
                    value = v.text == "1"
                elif kind in ("str", "e"):
                    value = v.text
                else:
                    try:
                        value = float(v.text)
                    except ValueError:
                        value = v.text
            while len(row) <= col:
                row.append(None)
            row[col] = value
    return rows, names


def _pick_sheet(names: list[str], sheet: Any, fname: str) -> int:
    if sheet is None:
        return 0
    if is_text(sheet):
        want = text(sheet)
        if want in names:
            return names.index(want)
        raise MatlabError(f"Worksheet '{want}' was not found in '{fname}'. Its sheets are: "
                          + ", ".join(names) + ".")
    k = int(np.asarray(sheet).flat[0])
    if not 1 <= k <= len(names):
        raise MatlabError(f"'{fname}' has {len(names)} worksheet(s); sheet number {k} does not exist.")
    return k - 1


def _openpyxl_read(openpyxl, fname: str, sheet: Any) -> tuple[list[list], list[str]]:
    try:
        wb = openpyxl.load_workbook(fname, data_only=True, read_only=True)
    except Exception as e:  # noqa: BLE001 - openpyxl raises several unrelated error types
        raise MatlabError(f"'{fname}' is not an Excel workbook that can be read ({e}).") from None
    names = list(wb.sheetnames)
    ws = wb[names[_pick_sheet(names, sheet, fname)]]
    rows = []
    for row in ws.iter_rows(values_only=True):
        out = []
        for v in row:
            if isinstance(v, bool) or v is None or isinstance(v, str):
                out.append(v)
            elif isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, _dt.datetime):
                out.append((v - _EPOCH).total_seconds() / 86400.0)
            elif isinstance(v, _dt.date):
                out.append(float((_dt.datetime(v.year, v.month, v.day) - _EPOCH).days))
            elif isinstance(v, _dt.time):
                out.append((v.hour * 3600 + v.minute * 60 + v.second) / 86400.0)
            else:
                out.append(str(v))
        rows.append(out)
    wb.close()
    return rows, names


def read_sheet(fname: str, sheet: Any = None, rng: str | None = None, what: str = "xlsread") -> list[list]:
    """The used cells of a worksheet as rows of float, str, bool or None (rectangular)."""
    if not os.path.isfile(fname):
        raise MatlabError(f"{what}: the file '{fname}' was not found.")
    _check_format(fname, what)
    lib = _openpyxl()
    rows, _ = _openpyxl_read(lib, fname, sheet) if lib else _builtin_read(fname, sheet)
    r0, c0, r1, c1 = _parse_range(rng, what)
    if rng:
        last_r = r1 if r1 is not None else len(rows) - 1
        rows = [list(r[c0:(c1 + 1 if c1 is not None else None)]) for r in rows[r0:last_r + 1]]
        if r1 is not None:
            rows += [[] for _ in range(r1 - r0 + 1 - len(rows))]
        width = (c1 - c0 + 1) if c1 is not None else max((len(r) for r in rows), default=0)
    else:
        # Drop empty rows and columns around the data.
        while rows and all(v is None for v in rows[-1]):
            rows.pop()
        while rows and all(v is None for v in rows[0]):
            rows.pop(0)
        width = max((max((k + 1 for k, v in enumerate(r) if v is not None), default=0) for r in rows),
                    default=0)
        lead = min((next((k for k, v in enumerate(r) if v is not None), width) for r in rows), default=0)
        rows = [list(r[lead:width]) for r in rows]
        width -= lead
    return [r + [None] * (width - len(r)) for r in rows]


def sheet_names(fname: str, what: str) -> list[str]:
    if not os.path.isfile(fname):
        raise MatlabError(f"{what}: the file '{fname}' was not found.")
    _check_format(fname, what)
    lib = _openpyxl()
    if lib:
        return _openpyxl_read(lib, fname, None)[1]
    return _builtin_read(fname, None)[1]


# -- writing -----------------------------------------------------------------------------------------

def _cell_xml(ref: str, v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return f'<c r="{ref}" t="b"><v>{int(v)}</v></c>'
    if isinstance(v, str):
        keep = ' xml:space="preserve"' if v != v.strip() else ""
        return f'<c r="{ref}" t="inlineStr"><is><t{keep}>{escape(v)}</t></is></c>'
    if np.isnan(v) or np.isinf(v):
        return ""
    return f'<c r="{ref}"><v>{repr(float(v)) if v != int(v) else int(v)}</v></c>'


def _sheet_xml(rows: list[list]) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             f'<worksheet xmlns="{_NS["m"]}"><sheetData>']
    for r, row in enumerate(rows):
        cells = "".join(_cell_xml(f"{_col_letters(c)}{r + 1}", v) for c, v in enumerate(row))
        if cells:
            parts.append(f'<row r="{r + 1}">{cells}</row>')
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def _builtin_write(fname: str, sheets: list[tuple[str, list[list]]]) -> None:
    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.'
          'spreadsheetml.sheet.main+xml"/>']
    wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          f'<workbook xmlns="{_NS["m"]}" xmlns:r="{_NS["r"]}"><sheets>']
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', f'<Relationships xmlns="{_NS["p"]}">']
    for k, (name, _) in enumerate(sheets, start=1):
        ct.append(f'<Override PartName="/xl/worksheets/sheet{k}.xml" ContentType="application/vnd.'
                  'openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        wb.append(f'<sheet name="{escape(name, {chr(34): "&quot;"})}" sheetId="{k}" r:id="rId{k}"/>')
        rels.append(f'<Relationship Id="rId{k}" Type="{_NS["r"]}/worksheet" Target="worksheets/sheet{k}.xml"/>')
    ct.append("</Types>")
    wb.append("</sheets></workbook>")
    rels.append("</Relationships>")
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 f'<Relationships xmlns="{_NS["p"]}"><Relationship Id="rId1" Type="{_NS["r"]}/officeDocument" '
                 'Target="xl/workbook.xml"/></Relationships>')
    with zipfile.ZipFile(fname, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", "".join(wb))
        z.writestr("xl/_rels/workbook.xml.rels", "".join(rels))
        for k, (_, rows) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{k}.xml", _sheet_xml(rows))


def write_sheet(fname: str, rows: list[list], sheet: Any = None, rng: str | None = None,
                what: str = "xlswrite", replace_file: bool = False) -> None:
    """Write rows (float, str, bool or None) into a worksheet, keeping the file's other sheets."""
    _check_format(fname, what)
    if os.path.splitext(fname)[1].lower() not in (".xlsx", ".xlsm"):
        raise MatlabError(f"{what}: Excel files are written as .xlsx; '{fname}' has another extension.")
    r0, c0, _, _ = _parse_range(rng, what)
    lib = _openpyxl()
    exists = os.path.isfile(fname) and not replace_file
    if lib:
        wb = lib.load_workbook(fname) if exists else lib.Workbook()
        if is_text(sheet):
            name = text(sheet)
            ws = wb[name] if name in wb.sheetnames else (
                wb.active if not exists and len(wb.sheetnames) == 1 else wb.create_sheet(name))
            ws.title = name
        elif sheet is not None:
            k = int(np.asarray(sheet).flat[0])
            while len(wb.sheetnames) < k:
                wb.create_sheet()
            ws = wb[wb.sheetnames[k - 1]]
        else:
            ws = wb[wb.sheetnames[0]]
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    v = None
                ws.cell(row=r0 + r + 1, column=c0 + c + 1, value=v)
        wb.save(fname)
        return
    sheets: list[tuple[str, list[list]]] = []
    if exists:
        with zipfile.ZipFile(fname) as z:
            names = [n for n, _ in _builtin_sheets(z)]
        sheets = [(n, _builtin_read(fname, char(n))[0]) for n in names]
    names = [n for n, _ in sheets]
    if is_text(sheet):
        target = text(sheet)
    elif sheet is not None:
        k = int(np.asarray(sheet).flat[0])
        while len(sheets) < k:
            sheets.append((f"Sheet{len(sheets) + 1}", []))
        target = sheets[k - 1][0]
    else:
        target = names[0] if names else "Sheet1"
    if target not in [n for n, _ in sheets]:
        sheets.append((target, []))
    grid = next(g for n, g in sheets if n == target)
    for r, row in enumerate(rows):
        while len(grid) <= r0 + r:
            grid.append([])
        line = grid[r0 + r]
        while len(line) < c0 + len(row):
            line.append(None)
        for c, v in enumerate(row):
            line[c0 + c] = v
    _builtin_write(fname, sheets)


# -- conversions between sheet rows and MATLAB values ----------------------------------------------------

def _is_number(v: Any) -> bool:
    return isinstance(v, float) or isinstance(v, bool)


def rows_to_numeric(rows: list[list]) -> np.ndarray:
    """The block of numbers, with NaN where a cell holds text or nothing; header rows/columns dropped."""
    if not rows:
        return np.zeros((0, 0))
    grid = np.array([[float(v) if _is_number(v) else np.nan for v in r] for r in rows], dtype=float)
    grid = grid.reshape(len(rows), -1)
    has = ~np.isnan(grid)
    if not has.any():
        return np.zeros((0, 0))
    rr, cc = np.flatnonzero(has.any(axis=1)), np.flatnonzero(has.any(axis=0))
    return grid[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1]


def rows_to_cell(rows: list[list], missing: Any = None) -> Cell:
    data = np.empty((len(rows), len(rows[0]) if rows else 0), dtype=object)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            if v is None:
                data[r, c] = np.array([[np.nan]]) if missing is None else missing
            elif isinstance(v, bool):
                data[r, c] = np.array([[v]])
            elif isinstance(v, str):
                data[r, c] = char(v)
            else:
                data[r, c] = np.array([[float(v)]])
    return Cell(data)


def _rows_to_text(rows: list[list]) -> Cell:
    texts = [[v if isinstance(v, str) else None for v in r] for r in rows]
    last_r = max((r for r, row in enumerate(texts) if any(v is not None for v in row)), default=-1)
    last_c = max((c for row in texts for c, v in enumerate(row) if v is not None), default=-1)
    data = np.empty((last_r + 1, last_c + 1), dtype=object)
    for r in range(last_r + 1):
        for c in range(last_c + 1):
            data[r, c] = char(texts[r][c] or "")
    return Cell(data)


def value_to_rows(v: Any, what: str) -> list[list]:
    """Rows of plain values for a numeric/logical matrix, text, a cell array or a string array."""
    if isinstance(v, MString):
        return [[str(v)]]
    if is_char(v):
        return [["".join(row)] for row in v] if v.ndim == 2 and v.shape[0] > 1 else [[to_str(v)]]

    def plain(x: Any) -> Any:
        if isinstance(x, MString):
            return str(x)
        if is_char(x):
            return to_str(x)
        if isinstance(x, np.ndarray) and x.size == 1 and x.dtype.kind in "biuf":
            return bool(x.flat[0]) if x.dtype == bool else float(x.flat[0])
        if isinstance(x, np.ndarray) and x.size == 0:
            return None
        if x is None:
            return None
        raise MatlabError(f"{what}: each cell must hold one number, a logical value or text.")

    if isinstance(v, Cell):
        if v.data.ndim > 2:
            raise MatlabError(f"{what}: the cell array must have two dimensions.")
        return [[plain(x) for x in row] for row in v.data]
    if isinstance(v, StringArray):
        return [[x for x in row] for row in v.data.reshape(v.data.shape[0], -1)]
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        if v.ndim > 2:
            raise MatlabError(f"{what}: the array must have two dimensions.")
        if v.dtype == bool:
            return [[bool(x) for x in row] for row in v]
        return [[float(x) for x in row] for row in v]
    raise MatlabError(f"{what}: the data must be a numeric array, text or a cell array.")


def _sheet_and_range(rest: list, what: str) -> tuple[Any, str | None]:
    """xlsread/xlswrite positional (sheet, range): a lone 'A1:B2'-like text is a range."""
    sheet, rng = None, None
    vals = [a for a in rest if not (isinstance(a, np.ndarray) and a.size == 0 and not is_char(a))]
    if len(vals) >= 2:
        sheet, rng = vals[0], text(vals[1], "range")
    elif len(vals) == 1:
        a = vals[0]
        if is_text(a) and re.fullmatch(r"\$?[A-Za-z]{1,3}\$?\d+(:\$?[A-Za-z]{1,3}\$?\d+)?", text(a)):
            rng = text(a)
        else:
            sheet = a
    if rng == "":
        rng = None
    return sheet, rng


# -- xlsread, xlswrite, sheetnames ---------------------------------------------------------------------

@builtin("xlsread")
def _xlsread(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("xlsread needs a file name, e.g. [num, txt, raw] = xlsread('data.xlsx').")
    fname = text(args[0])
    if os.path.splitext(fname)[1].lower() in (".csv", ".txt"):
        num = BUILTINS["readmatrix"](interp, [args[0]], 1)
        return num, Cell.empty(0, 0), Cell.empty(0, 0)
    sheet, rng = _sheet_and_range(list(args[1:3]), "xlsread")
    rows = read_sheet(fname, sheet, rng, "xlsread")
    return rows_to_numeric(rows), _rows_to_text(rows), rows_to_cell(rows)


@builtin("xlswrite")
def _xlswrite(interp, args, nargout):
    if len(args) < 2 or not is_text(args[0]):
        raise MatlabError("xlswrite needs a file name and the data, e.g. xlswrite('out.xlsx', A).")
    fname = text(args[0])
    if not os.path.splitext(fname)[1]:
        fname += ".xlsx"
    sheet, rng = _sheet_and_range(list(args[2:4]), "xlswrite")
    write_sheet(fname, value_to_rows(args[1], "xlswrite"), sheet, rng, "xlswrite")
    if nargout:
        return np.array([[True]])
    return None


@builtin("sheetnames")
def _sheetnames(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("sheetnames needs the name of an Excel file.")
    names = sheet_names(text(args[0]), "sheetnames")
    data = np.empty((len(names), 1), dtype=object)
    for k, n in enumerate(names):
        data[k, 0] = n
    return StringArray(data)


@builtin("xlsfinfo")
def _xlsfinfo(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("xlsfinfo needs the name of an Excel file.")
    names = sheet_names(text(args[0]), "xlsfinfo")
    return char("Microsoft Excel Spreadsheet"), Cell.from_list([char(n) for n in names]), char("xlOpenXMLWorkbook")


# -- Excel files in readtable, readmatrix, readcell and their write counterparts ----------------------

def _file_options(args: list, names: tuple[str, ...]) -> dict:
    opts: dict = {}
    k = 0
    while k + 1 < len(args):
        if is_text(args[k]) and text(args[k]).lower() in names:
            opts[text(args[k]).lower()] = args[k + 1]
            k += 2
        else:
            k += 1
    return opts


def _flag(opts: dict, name: str, default: bool) -> bool:
    if name not in opts:
        return default
    v = opts[name]
    return text(v).lower() in ("on", "true") if is_text(v) else bool(np.asarray(v).flat[0])


def _range_opt(opts: dict) -> str | None:
    return text(opts["range"]) if "range" in opts and is_text(opts["range"]) else None


def _read_table(interp, args, nargout):
    from .table import TableObject
    fname = text(args[0])
    opts = _file_options(list(args[1:]), ("sheet", "range", "readvariablenames", "texttype",
                                          "readrownames", "variablenamingrule"))
    rows = read_sheet(fname, opts.get("sheet"), _range_opt(opts), "readtable")
    if not rows:
        return TableObject([], [])
    first_is_header = all(isinstance(v, str) or v is None for v in rows[0]) and any(
        isinstance(v, str) for v in rows[0])
    header = _flag(opts, "readvariablenames", first_is_header)
    body = rows[1:] if header else rows
    names: list[str] = []
    for k in range(len(rows[0])):
        raw = rows[0][k] if header and isinstance(rows[0][k], str) else f"Var{k + 1}"
        clean = re.sub(r"\W", "_", raw.strip()) or f"Var{k + 1}"
        if not clean[0].isalpha():
            clean = "x" + clean
        while clean in names:
            clean += "_1"
        names.append(clean)
    as_string = "texttype" in opts and text(opts["texttype"]).lower() == "string"
    cols = []
    for k in range(len(names)):
        col = [r[k] for r in body]
        if all(v is None or (isinstance(v, float)) for v in col):
            cols.append(np.array([np.nan if v is None else v for v in col], dtype=float).reshape(-1, 1))
        elif all(isinstance(v, bool) for v in col):
            cols.append(np.array(col, dtype=bool).reshape(-1, 1))
        else:
            d = np.empty((len(col), 1), dtype=object)
            for i, v in enumerate(col):
                s = "" if v is None else (v if isinstance(v, str) else f"{float(v):.15g}")
                d[i, 0] = s if as_string else char(s)
            cols.append(StringArray(d) if as_string else Cell(d))
    table = TableObject(names, cols)
    if _flag(opts, "readrownames", False) and cols and isinstance(cols[0], (Cell, StringArray)):
        row_names = [x if isinstance(x, str) else to_str(x) for x in cols[0].data.flatten()]
        table = TableObject(names[1:], cols[1:], row_names=row_names)
    return table


def _write_table(interp, args, nargout):
    t = args[0]
    fname = text(args[1])
    opts = _file_options(list(args[2:]), ("sheet", "range", "writevariablenames", "writerownames",
                                          "writemode"))
    header: list = []
    columns: list[list[list]] = []
    for name, c in zip(t.names, t.columns):
        if isinstance(c, Cell):
            block = [[_plain_table_cell(x)] for x in c.data[:, 0]]
        elif isinstance(c, StringArray):
            block = [[x] for x in c.data[:, 0]]
        elif is_char(c):
            block = [["".join(row)] for row in c]
        elif isinstance(c, np.ndarray):
            block = [[bool(v) if c.dtype == bool else float(v) for v in row] for row in c.reshape(c.shape[0], -1)]
        elif hasattr(c, "element_texts"):
            block = [[str(s)] for s in np.asarray(c.element_texts()).flatten(order="F")]
        else:
            raise MatlabError(f"writetable: the variable '{name}' cannot be written to an Excel file.")
        width = len(block[0]) if block else 1
        header += [name] if width == 1 else [f"{name}_{k + 1}" for k in range(width)]
        columns.append(block)
    rows = [[v for block in columns for v in block[i]] for i in range(t.nrows)]
    if _flag(opts, "writerownames", False) and t.row_names:
        rows = [[n] + r for n, r in zip(t.row_names, rows)]
        header = ["Row"] + header
    if _flag(opts, "writevariablenames", True):
        rows = [header] + rows
    replace = "writemode" in opts and text(opts["writemode"]).lower() == "replacefile"
    write_sheet(fname, rows, opts.get("sheet"), _range_opt(opts), "writetable", replace_file=replace)
    return None


def _plain_table_cell(x: Any) -> Any:
    if is_char(x) or isinstance(x, MString):
        return to_str(x)
    if isinstance(x, np.ndarray) and x.size == 1:
        return bool(x.flat[0]) if x.dtype == bool else float(x.flat[0])
    if isinstance(x, np.ndarray) and x.size == 0:
        return None
    return " ".join(f"{v:.15g}" for v in np.asarray(x, dtype=float).flatten())


def _read_matrix(interp, args, nargout):
    opts = _file_options(list(args[1:]), ("sheet", "range", "numheaderlines", "outputtype"))
    rows = read_sheet(text(args[0]), opts.get("sheet"), _range_opt(opts), "readmatrix")
    if "numheaderlines" in opts:
        rows = rows[int(np.asarray(opts["numheaderlines"]).flat[0]):]
    return rows_to_numeric(rows)


def _read_cell(interp, args, nargout):
    opts = _file_options(list(args[1:]), ("sheet", "range", "numheaderlines"))
    rows = read_sheet(text(args[0]), opts.get("sheet"), _range_opt(opts), "readcell")
    if "numheaderlines" in opts:
        rows = rows[int(np.asarray(opts["numheaderlines"]).flat[0]):]
    d = np.empty((1, 1), dtype=object)
    d[0, 0] = None
    return rows_to_cell(rows, missing=StringArray(d))


def _write_values(name: str):
    def fn(interp, args, nargout):
        opts = _file_options(list(args[2:]), ("sheet", "range", "writemode"))
        if name == "writecell" and not isinstance(args[0], Cell):
            raise MatlabError("writecell expects a cell array.")
        replace = "writemode" in opts and text(opts["writemode"]).lower() == "replacefile"
        write_sheet(text(args[1]), value_to_rows(args[0], name), opts.get("sheet"), _range_opt(opts), name,
                    replace_file=replace)
        return None
    return fn


def _route(name: str, excel_fn, file_arg: int) -> None:
    """Send calls with an Excel file name to ``excel_fn``; everything else to the original function."""
    original = BUILTINS.get(name)
    if original is None:
        return

    def wrapper(interp, args, nargout):
        if len(args) > file_arg and is_excel_file(args[file_arg]):
            return excel_fn(interp, args, nargout)
        return original(interp, args, nargout)

    wrapper.__name__ = getattr(original, "__name__", name)
    wrapper.__doc__ = original.__doc__
    BUILTINS[name] = wrapper


_route("readtable", _read_table, 0)
_route("writetable", _write_table, 1)
_route("readmatrix", _read_matrix, 0)
_route("readcell", _read_cell, 0)
_route("writematrix", _write_values("writematrix"), 1)
_route("writecell", _write_values("writecell"), 1)
