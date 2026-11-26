"""OpenDocument spreadsheets (.ods): a zip archive whose content.xml holds the sheets.

Written from the OpenDocument 1.2 specification (OASIS). Cells come back as float, str, bool or
None like the .xlsx reader; dates become Excel-style serial day numbers and times fractions of a
day, so the spreadsheet readers treat every workbook format the same way.
"""

from __future__ import annotations

import datetime as _dt
import re
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from ..errors import MatlabError

_OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_EPOCH = _dt.datetime(1899, 12, 30)
_MIME = "application/vnd.oasis.opendocument.spreadsheet"
_MAX_REPEAT = 1 << 20  # a filled row or cell repeated more often than this is taken as a formatting artefact


def _o(name: str) -> str:
    return f"{{{_OFFICE}}}{name}"


def _t(name: str) -> str:
    return f"{{{_TABLE}}}{name}"


def _x(name: str) -> str:
    return f"{{{_TEXT}}}{name}"


def _content(fname: str) -> ET.Element:
    try:
        with zipfile.ZipFile(fname) as z:
            return ET.fromstring(z.read("content.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        raise MatlabError(f"'{fname}' is not an OpenDocument spreadsheet (.ods) that can be read.") from None


def _tables(root: ET.Element) -> list[ET.Element]:
    body = root.find(_o("body"))
    sheet = body.find(_o("spreadsheet")) if body is not None else None
    return [] if sheet is None else sheet.findall(_t("table"))


def _paragraph_text(el: ET.Element) -> str:
    parts = [el.text or ""]
    for child in el:
        if child.tag == _x("s"):
            parts.append(" " * int(child.get(_x("c"), "1")))
        elif child.tag == _x("tab"):
            parts.append("\t")
        elif child.tag == _x("line-break"):
            parts.append("\n")
        else:
            parts.append(_paragraph_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _date_serial(s: str) -> float | str:
    try:
        d = _dt.datetime.fromisoformat(s[:26])
    except ValueError:
        return s
    if d.tzinfo is not None:
        d = d.replace(tzinfo=None)
    return (d - _EPOCH).total_seconds() / 86400.0


def _time_fraction(s: str) -> float | str:
    m = re.fullmatch(r"-?P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", s.strip())
    if not m:
        return s
    days, hours, mins, secs = (float(g) if g else 0.0 for g in m.groups())
    value = days + (hours * 3600 + mins * 60 + secs) / 86400.0
    return -value if s.strip().startswith("-") else value


def _cell_value(c: ET.Element) -> Any:
    kind = c.get(_o("value-type"))
    if kind in ("float", "percentage", "currency"):
        try:
            return float(c.get(_o("value"), "nan"))
        except ValueError:
            return None
    if kind == "boolean":
        return c.get(_o("boolean-value"), "false").lower() == "true"
    if kind == "date":
        return _date_serial(c.get(_o("date-value"), ""))
    if kind == "time":
        return _time_fraction(c.get(_o("time-value"), ""))
    paras = c.findall(_x("p"))
    if kind == "string" or paras:
        value = c.get(_o("string-value"))
        return value if value is not None else "\n".join(_paragraph_text(p) for p in paras)
    return None


def _rows_of(table: ET.Element) -> list[ET.Element]:
    """The table-row elements of a sheet, looking inside header-row and row-group wrappers."""
    out = []
    for el in table:
        if el.tag == _t("table-row"):
            out.append(el)
        elif el.tag in (_t("table-header-rows"), _t("table-row-group"), _t("table-rows")):
            out += _rows_of(el)
    return out


def _sheet_rows(table: ET.Element) -> list[list]:
    rows: list[list] = []
    r = 0
    for row_el in _rows_of(table):
        repeat = int(row_el.get(_t("number-rows-repeated"), "1"))
        line: list = []
        col = 0
        for c in row_el:
            if c.tag not in (_t("table-cell"), _t("covered-table-cell")):
                continue
            n = int(c.get(_t("number-columns-repeated"), "1"))
            value = _cell_value(c)
            if value is not None:
                if n > _MAX_REPEAT:
                    n = 1
                line += [None] * (col - len(line)) + [value] * n
            col += n
        if line:
            if repeat > _MAX_REPEAT:
                repeat = 1
            rows += [[] for _ in range(r - len(rows))]
            rows += [list(line) for _ in range(repeat)]
        r += repeat
    return rows


def read(fname: str, sheet: Any, pick) -> tuple[list[list], list[str]]:
    """Rows of one sheet (picked by ``pick(names, sheet, fname)``) and the names of all sheets."""
    tables = _tables(_content(fname))
    names = [t.get(_t("name"), f"Sheet{k + 1}") for k, t in enumerate(tables)]
    if not tables:
        raise MatlabError(f"'{fname}' holds no spreadsheet tables.")
    return _sheet_rows(tables[pick(names, sheet, fname)]), names


# -- writing ---------------------------------------------------------------------------------------------

def _esc(s: str) -> str:
    from xml.sax.saxutils import escape
    return escape(s, {'"': "&quot;"})


def _cell_xml(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "<table:table-cell/>"
    if isinstance(v, bool):
        word = "true" if v else "false"
        return (f'<table:table-cell office:value-type="boolean" office:boolean-value="{word}">'
                f"<text:p>{word.upper()}</text:p></table:table-cell>")
    if isinstance(v, str):
        lines = "".join(f"<text:p>{_esc(line)}</text:p>" for line in v.split("\n"))
        return f'<table:table-cell office:value-type="string">{lines}</table:table-cell>'
    x = float(v)
    shown = repr(x) if x != int(x) else str(int(x))
    return (f'<table:table-cell office:value-type="float" office:value="{shown}">'
            f"<text:p>{shown}</text:p></table:table-cell>")


def write(fname: str, sheets: list[tuple[str, list[list]]]) -> None:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<office:document-content xmlns:office="{_OFFICE}" xmlns:table="{_TABLE}" xmlns:text="{_TEXT}" '
             'office:version="1.2"><office:body><office:spreadsheet>']
    for name, rows in sheets:
        width = max((len(r) for r in rows), default=0)
        parts.append(f'<table:table table:name="{_esc(name)}">')
        parts.append(f'<table:table-column table:number-columns-repeated="{max(width, 1)}"/>')
        for row in rows:
            parts.append("<table:table-row>" + "".join(_cell_xml(v) for v in row) + "</table:table-row>")
        if not rows:
            parts.append("<table:table-row><table:table-cell/></table:table-row>")
        parts.append("</table:table>")
    parts.append("</office:spreadsheet></office:body></office:document-content>")
    manifest = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" '
                'manifest:version="1.2">'
                f'<manifest:file-entry manifest:full-path="/" manifest:media-type="{_MIME}"/>'
                '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
                '</manifest:manifest>')
    with zipfile.ZipFile(fname, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), _MIME, compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/manifest.xml", manifest)
        z.writestr("content.xml", "".join(parts))
