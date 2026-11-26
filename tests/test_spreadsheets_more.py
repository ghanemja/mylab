"""OpenDocument (.ods) and Excel 97-2003 (.xls) spreadsheets, and the refusal of .xlsb.

The test files are made here: .ods archives with zipfile, and .xls workbooks with the small BIFF8
writer below (an OLE2 compound file holding a Workbook stream), written from the published
[MS-CFB] and [MS-XLS] layouts.
"""

import struct
import zipfile

import numpy as np
import pytest

from mylab.builtins import _xls, excel
from mylab.errors import MatlabError


@pytest.fixture
def d(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return m


# -- a minimal .ods writer ---------------------------------------------------------------------------

_ODS_HEAD = ('<?xml version="1.0" encoding="UTF-8"?><office:document-content '
             'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
             'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
             'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:spreadsheet>')
_ODS_TAIL = "</office:spreadsheet></office:body></office:document-content>"


def _ods(path, tables: dict) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        body = "".join(f'<table:table table:name="{name}">{rows}</table:table>' for name, rows in tables.items())
        z.writestr("content.xml", _ODS_HEAD + body + _ODS_TAIL)


def _num(x):
    return f'<table:table-cell office:value-type="float" office:value="{x}"><text:p>{x}</text:p></table:table-cell>'


def _str(s):
    return f'<table:table-cell office:value-type="string"><text:p>{s}</text:p></table:table-cell>'


def _row(*cells, repeat=1):
    extra = f' table:number-rows-repeated="{repeat}"' if repeat > 1 else ""
    return f"<table:table-row{extra}>" + "".join(cells) + "</table:table-row>"


_EMPTY = '<table:table-cell table:number-columns-repeated="{n}"/>'


@pytest.fixture
def ods_file(d, tmp_path):
    first = (_row(_str("name"), _str("score"), _str("passed"))
             + _row(_str("ann"), _num(9.5),
                    '<table:table-cell office:value-type="boolean" office:boolean-value="true">'
                    "<text:p>TRUE</text:p></table:table-cell>")
             + _row(_EMPTY.format(n=3), repeat=2)
             + _row(_str("bob"), _num(7), '<table:table-cell office:value-type="boolean" '
                    'office:boolean-value="false"><text:p>FALSE</text:p></table:table-cell>', _EMPTY.format(n=1000))
             + _row(_EMPTY.format(n=1024), repeat=1048570))
    second = ("<table:table-header-rows>" + _row(_num(1), _num(2)) + "</table:table-header-rows>"
              + _row('<table:table-cell office:value-type="float" office:value="5" '
                     'table:number-columns-repeated="2"><text:p>5</text:p></table:table-cell>', repeat=2)
              + _row('<table:table-cell office:value-type="date" office:date-value="2024-01-02T12:00:00">'
                     "<text:p>x</text:p></table:table-cell>",
                     '<table:table-cell office:value-type="time" office:time-value="PT06H00M00S">'
                     "<text:p>06:00</text:p></table:table-cell>")
              + _row('<table:table-cell office:value-type="string"><text:p>a<text:s text:c="2"/>b</text:p>'
                     "<text:p>line 2</text:p></table:table-cell>", _num(3)))
    _ods(tmp_path / "data.ods", {"People": first, "Numbers": second})
    return d


# -- .ods ---------------------------------------------------------------------------------------------------

def test_ods_readers(ods_file):
    m = ods_file
    m.run("names = sheetnames('data.ods'); T = readtable('data.ods'); C = readcell('data.ods');"
          "M = readmatrix('data.ods', 'Sheet', 'Numbers'); [num, txt, raw] = xlsread('data.ods', 'People');"
          "C2 = readcell('data.ods', 'Sheet', 2); part = readmatrix('data.ods', 'Sheet', 2, 'Range', 'A2:B3');")
    assert m.num("numel(names)") == 2 and m.num('names(2) == "Numbers"')
    assert m.val("T").names == ["name", "score", "passed"]
    assert m.num("height(T)") == 4  # the two blank rows are kept as empty rows
    np.testing.assert_array_equal(m.num("T.score([1 4])"), [[9.5], [7]])
    assert m.str("T.name{4}") == "bob"
    assert m.num("size(C)").tolist() == [[5, 3]] and m.str("C{1, 2}") == "score"
    assert m.num("C{2, 3}").dtype == bool and bool(m.num("C{2, 3}")) is True
    np.testing.assert_array_equal(m.num("M(1:3, :)"), [[1, 2], [5, 5], [5, 5]])
    assert m.num("M(4, 1)") == pytest.approx(45293.5) and m.num("M(4, 2)") == pytest.approx(0.25)
    np.testing.assert_array_equal(m.num("num"), [[9.5, 1], [np.nan, np.nan], [np.nan, np.nan], [7, 0]])
    assert m.str("txt{5, 1}") == "bob" and m.str("raw{1, 3}") == "passed"
    assert m.str("C2{5, 1}") == "a  b\nline 2"
    np.testing.assert_array_equal(m.num("part"), [[5, 5], [5, 5]])
    m.run("[~, sheets, fmt] = xlsfinfo('data.ods');")
    assert m.str("fmt") == "xlOpenDocumentSpreadsheet" and m.str("sheets{1}") == "People"


def test_ods_round_trip(d):
    m = d
    m.run("T = table([1; 2], {'x'; 'y'}, [true; false], 'VariableNames', {'id', 'tag', 'ok'});"
          "writetable(T, 'out.ods'); R = readtable('out.ods');"
          "writematrix([1.5 2; 3 4], 'out.ods', 'Sheet', 'M'); M = readmatrix('out.ods', 'Sheet', 'M');"
          "R2 = readtable('out.ods', 'Sheet', 1); writecell({'a', 1; 2, 'b & <c>'}, 'cells.ods'); "
          "C = readcell('cells.ods'); xlswrite('x.ods', magic(3), 2); X = xlsread('x.ods', 2);"
          "names = sheetnames('out.ods');")
    assert m.val("R").names == ["id", "tag", "ok"] and m.str("R.tag{2}") == "y"
    assert m.num("R.ok").dtype == bool
    np.testing.assert_array_equal(m.num("M"), [[1.5, 2], [3, 4]])
    assert m.num("height(R2)") == 2  # the first sheet is kept when another one is written
    assert m.str("C{2, 2}") == "b & <c>" and m.num("C{2, 1}") == 2
    np.testing.assert_array_equal(m.num("X"), [[8, 1, 6], [3, 5, 7], [4, 9, 2]])
    assert m.num("numel(names)") == 2
    with zipfile.ZipFile("out.ods") as z:
        assert z.namelist()[0] == "mimetype" and z.getinfo("mimetype").compress_type == zipfile.ZIP_STORED


def test_ods_errors(d, tmp_path):
    m = d
    (tmp_path / "bad.ods").write_bytes(b"not a zip")
    with pytest.raises(MatlabError, match="OpenDocument"):
        m.run("readmatrix('bad.ods')")
    _ods(tmp_path / "one.ods", {"S": _row(_num(1))})
    with pytest.raises(MatlabError, match="Worksheet 'Nope' was not found"):
        m.run("readmatrix('one.ods', 'Sheet', 'Nope')")
    with pytest.raises(MatlabError, match="NoSuchOption__"):
        m.run("readmatrix('one.ods', 'NoSuchOption__', 1)")


# -- a minimal BIFF8 (.xls) writer -------------------------------------------------------------------------

def _rec(kind: int, payload: bytes) -> bytes:
    return struct.pack("<HH", kind, len(payload)) + payload


def _ustr(s: str, count_bytes: int = 2) -> bytes:
    """An unrolled unicode string: count, a flags byte (wide when not Latin-1) and the characters."""
    wide = any(ord(ch) > 255 for ch in s)
    count = struct.pack("<H", len(s)) if count_bytes == 2 else bytes([len(s)])
    return count + bytes([1 if wide else 0]) + s.encode("utf-16-le" if wide else "latin-1")


def _bof(kind: int) -> bytes:
    return _rec(0x0809, struct.pack("<HHHHII", 0x0600, kind, 0x0DBB, 0x07CC, 0, 6))


def _rk_int(n: int, x100: bool = False) -> int:
    return ((n << 2) | 2 | (1 if x100 else 0)) & 0xFFFFFFFF


def _rk_float(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0] >> 32  # exact for values like 0.5 or 1.25


def _workbook(sheets: list[tuple[str, bytes]], sst_records: bytes = b"") -> bytes:
    """The Workbook stream: globals (BOF, shared strings, sheet list, EOF), then one substream per sheet."""
    def globals_part(offsets):
        bound = b"".join(_rec(0x0085, struct.pack("<IBB", off, 0, 0) + _ustr(name, 1))
                         for (name, _), off in zip(sheets, offsets))
        return _bof(5) + _rec(0x0022, struct.pack("<H", 0)) + sst_records + bound + _rec(0x000A, b"")
    head = len(globals_part([0] * len(sheets)))
    offsets, pos = [], head
    for _, body in sheets:
        offsets.append(pos)
        pos += len(_bof(0x10)) + len(body) + 4
    return globals_part(offsets) + b"".join(_bof(0x10) + body + _rec(0x000A, b"") for _, body in sheets)


def _compound(stream: bytes, mini: bool = False) -> bytes:
    """An OLE2 compound file (version 3, 512-byte sectors) with one stream called Workbook.

    Sector 0 holds the FAT, sector 1 the directory; a large stream (4096 bytes or more) fills the
    sectors after them. With ``mini`` the stream stays small and lives in the mini stream: sector 2
    holds the mini FAT and the sectors after it the mini stream (owned by the root entry).
    """
    end, free, fatsect, nostream = 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFD, 0xFFFFFFFF
    if not mini:
        stream = stream + b"\0" * max(0, 4096 - len(stream))
    body = stream + b"\0" * (-len(stream) % (64 if mini else 512))
    fat = [fatsect, end]
    if mini:
        n_mini_sectors = len(body) // 64
        minifat = [k + 1 for k in range(n_mini_sectors - 1)] + [end]
        minifat_bytes = struct.pack(f"<{len(minifat)}I", *minifat)
        minifat_bytes += b"\xff" * (512 - len(minifat_bytes))
        fat.append(end)  # sector 2: the mini FAT
        data_start = 3
        stored = body + b"\0" * (-len(body) % 512)
    else:
        data_start = 2
        stored = body
    n_data = len(stored) // 512
    fat += [data_start + k + 1 for k in range(n_data - 1)] + [end]
    assert len(fat) <= 128
    fat += [free] * (128 - len(fat))

    def entry(name, kind, child, start, size):
        raw = name.encode("utf-16-le") + b"\0\0"
        return (raw + b"\0" * (64 - len(raw)) + struct.pack("<HBB", len(raw), kind, 1)
                + struct.pack("<III", nostream, nostream, child) + b"\0" * 16 + b"\0" * 4 + b"\0" * 16
                + struct.pack("<IQ", start, size))

    if mini:
        root = entry("Root Entry", 5, 1, data_start, len(body))
        book = entry("Workbook", 2, nostream, 0, len(stream))
    else:
        root = entry("Root Entry", 5, 1, end, 0)
        book = entry("Workbook", 2, nostream, data_start, len(stream))
    directory = root + book + entry("", 0, nostream, end, 0) * 2
    header = (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 16 + struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6)
              + b"\0" * 6 + struct.pack("<IIIIIIIII", 0, 1, 1, 0, 4096, 2 if mini else end, 1 if mini else 0,
                                        end, 0)
              + struct.pack("<109I", 0, *([free] * 108)))
    assert len(header) == 512
    sectors = struct.pack("<128I", *fat) + directory + (minifat_bytes if mini else b"") + stored
    return header + sectors


def _cell(kind: int, r: int, c: int, rest: bytes) -> bytes:
    return _rec(kind, struct.pack("<HHH", r, c, 15) + rest)


def _sst() -> bytes:
    """Shared strings; the third string is split by a CONTINUE record and goes on in 16-bit characters."""
    strings = [_ustr("name"), _ustr("value"), _ustr("wörd π")]
    head = struct.pack("<II", 6, 3)
    third = strings[2]
    cut = 3 + 0  # the third string is wide: count, flags, then split after two characters
    first_part = head + strings[0] + strings[1] + third[:cut + 4]
    rest = b"\x01" + third[cut + 4:]
    return _rec(0x00FC, first_part) + _rec(0x003C, rest)


def _formula(r: int, c: int, result: bytes) -> bytes:
    return _cell(0x0006, r, c, result + struct.pack("<HI", 0, 0) + struct.pack("<H", 0))


def _xls_bytes(mini: bool = False) -> bytes:
    sheet1 = (_cell(0x00FD, 0, 0, struct.pack("<I", 0)) + _cell(0x00FD, 0, 1, struct.pack("<I", 1))
              + _cell(0x0204, 1, 0, _ustr("alpha")) + _cell(0x0203, 1, 1, struct.pack("<d", 2.5))
              + _cell(0x00FD, 2, 0, struct.pack("<I", 2)) + _cell(0x027E, 2, 1, struct.pack("<I", _rk_int(-7)))
              + _cell(0x0205, 3, 0, bytes([1, 0])) + _cell(0x027E, 3, 1, struct.pack("<I", _rk_int(1234, True)))
              + _rec(0x00BD, struct.pack("<HH", 4, 0) + struct.pack("<HI", 15, _rk_float(0.5))
                     + struct.pack("<HI", 15, _rk_float(1.25)) + struct.pack("<H", 1))
              + _formula(5, 0, struct.pack("<d", 42.0))
              + _formula(5, 1, bytes([1, 0, 1, 0, 0, 0, 0xFF, 0xFF])))
    sheet2 = (_formula(0, 0, bytes([0, 0, 0, 0, 0, 0, 0xFF, 0xFF])) + _rec(0x0207, _ustr("from formula"))
              + _cell(0x0205, 0, 1, bytes([0x07, 1])) + _cell(0x0203, 2, 2, struct.pack("<d", -1e-3)))
    return _compound(_workbook([("Data", sheet1), ("Other", sheet2)], _sst()), mini)


@pytest.fixture(params=[False, True], ids=["sectors", "ministream"])
def xls_file(request, d, tmp_path):
    (tmp_path / "book.xls").write_bytes(_xls_bytes(request.param))
    return d


def test_xls_reader(xls_file):
    m = xls_file
    m.run("names = sheetnames('book.xls'); C = readcell('book.xls'); C2 = readcell('book.xls', 'Sheet', 'Other');"
          "[num, txt, raw] = xlsread('book.xls'); T = readtable('book.xls'); M = readmatrix('book.xls', 'Sheet', 2);")
    assert m.num("numel(names)") == 2 and m.num('names(1) == "Data"') and m.num('names(2) == "Other"')
    assert m.num("size(C)").tolist() == [[6, 2]]
    assert m.str("C{1, 1}") == "name" and m.str("C{2, 1}") == "alpha" and m.num("C{2, 2}") == 2.5
    assert m.str("C{3, 1}") == "wörd π" and m.num("C{3, 2}") == -7
    assert m.num("C{4, 1}").dtype == bool and m.num("C{4, 2}") == pytest.approx(12.34)
    assert m.num("C{5, 1}") == 0.5 and m.num("C{5, 2}") == 1.25
    assert m.num("C{6, 1}") == 42 and m.num("C{6, 2}").dtype == bool
    np.testing.assert_allclose(m.num("num"), [[np.nan, 2.5], [np.nan, -7], [1, 12.34], [0.5, 1.25], [42, 1]])
    assert m.str("txt{3, 1}") == "wörd π" and m.str("raw{1, 2}") == "value"
    assert m.val("T").names == ["name", "value"] and m.num("height(T)") == 5
    assert m.str("C2{1, 1}") == "from formula" and m.str("C2{1, 2}") == "#DIV/0!"
    assert m.num("M(end, end)") == -1e-3
    m.run("[~, sheets, fmt] = xlsfinfo('book.xls');")
    assert m.str("fmt") == "xlWorkbookNormal" and m.str("sheets{2}") == "Other"


def test_xls_errors(d, tmp_path):
    m = d
    (tmp_path / "plain.xls").write_text("a,b\n1,2\n")
    with pytest.raises(MatlabError, match="not supported.*BIFF8"):
        m.run("readmatrix('plain.xls')")
    (tmp_path / "empty.xls").write_bytes(_compound(b"\x09\x08\x10\x00" + b"\0" * 16, mini=True)[:600])
    with pytest.raises(MatlabError, match="not supported"):
        m.run("readcell('empty.xls')")
    (tmp_path / "book.xls").write_bytes(_xls_bytes())
    with pytest.raises(MatlabError, match="written as Excel workbooks"):
        m.run("writematrix(1, 'book.xls')")
    with pytest.raises(MatlabError, match="sheet number 3 does not exist"):
        m.run("xlsread('book.xls', 3)")


def test_xls_uses_xlrd_when_present(d, tmp_path, monkeypatch):
    """With xlrd importable it does the reading; the built-in reader is not called."""
    import sys
    import types
    fake = types.SimpleNamespace(XL_CELL_EMPTY=0, XL_CELL_TEXT=1, XL_CELL_NUMBER=2, XL_CELL_DATE=3,
                                 XL_CELL_BOOLEAN=4, XL_CELL_ERROR=5, XL_CELL_BLANK=6)
    cell = types.SimpleNamespace

    class Book:
        def sheet_names(self):
            return ["Only"]

        def sheet_by_index(self, k):
            return types.SimpleNamespace(nrows=1, row=lambda r: [cell(ctype=1, value="hi"), cell(ctype=2, value=3.0),
                                                                 cell(ctype=4, value=1)])

        def release_resources(self):
            pass

    fake.open_workbook = lambda name, on_demand=True: Book()
    monkeypatch.setitem(sys.modules, "xlrd", fake)
    monkeypatch.setattr(_xls, "_stream", lambda *a: pytest.fail("the built-in reader was used"))
    (tmp_path / "x.xls").write_bytes(b"")
    m = d
    m.run("C = readcell('x.xls');")
    assert m.str("C{1, 1}") == "hi" and m.num("C{1, 2}") == 3 and m.num("C{1, 3}").dtype == bool


def test_xlsb_is_refused_clearly(d, tmp_path):
    m = d
    (tmp_path / "b.xlsb").write_bytes(b"PK")
    for call in ("readtable('b.xlsb')", "xlsread('b.xlsb')", "sheetnames('b.xlsb')", "writematrix(1, 'b.xlsb')"):
        with pytest.raises(MatlabError, match=r"\.xlsb\) are not supported"):
            m.run(call)


def test_xls_reader_without_openpyxl(xls_file, monkeypatch):
    monkeypatch.setattr(excel, "_openpyxl", lambda: None)
    m = xls_file
    m.run("A = readmatrix('book.xls');")
    assert m.num("A(1, 2)") == 2.5
