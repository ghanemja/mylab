"""Legacy Excel workbooks (.xls): cell values from the BIFF8 records inside an OLE2 compound file.

Written from the published specifications ([MS-CFB] for the container, [MS-XLS] for the records).
Only what the spreadsheet readers need is decoded: sheet names, numbers (NUMBER, RK, MULRK), text
(shared strings, LABEL), logical and error values (BOOLERR), and the cached results of formulas.
Formatting is ignored, so dates come back as serial day numbers, as with the .xlsx reader. When
the optional ``xlrd`` package is installed it does the reading instead.
"""

from __future__ import annotations

import struct
from typing import Any

from ..errors import MatlabError

_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_END = 0xFFFFFFFE
_FREE = 0xFFFFFFFF
_ERRORS = {0x00: "#NULL!", 0x07: "#DIV/0!", 0x0F: "#VALUE!", 0x17: "#REF!", 0x1D: "#NAME?", 0x24: "#NUM!",
           0x2A: "#N/A", 0x2B: "#GETTING_DATA"}


class _Bad(Exception):
    pass


# -- the compound file ---------------------------------------------------------------------------------

def _chain(table: list[int], start: int, limit: int) -> list[int]:
    out = []
    while start not in (_END, _FREE) and len(out) <= limit:
        if start >= len(table):
            raise _Bad("a sector chain runs past the allocation table")
        out.append(start)
        start = table[start]
    return out


def _stream(data: bytes, name: str) -> bytes | None:
    """The contents of the stream called ``name`` (case-insensitive), or None."""
    if len(data) < 512 or data[:8] != _SIGNATURE:
        raise _Bad("not a compound file")
    shift, mini_shift = struct.unpack_from("<HH", data, 0x1E)
    size, mini_size = 1 << shift, 1 << mini_shift
    n_fat, first_dir = struct.unpack_from("<II", data, 0x2C)
    cutoff, first_mini, n_mini, first_difat, n_difat = struct.unpack_from("<IIIII", data, 0x38)

    def sector(k: int) -> bytes:
        off = (k + 1) * size
        if off >= len(data):
            raise _Bad("a sector lies beyond the end of the file")
        return data[off:off + size]

    difat = list(struct.unpack_from("<109I", data, 0x4C))
    k = first_difat
    for _ in range(n_difat):
        if k in (_END, _FREE):
            break
        entries = struct.unpack(f"<{size // 4}I", sector(k))
        difat += entries[:-1]
        k = entries[-1]
    fat: list[int] = []
    for s in [d for d in difat if d not in (_END, _FREE)][:n_fat]:
        fat += struct.unpack(f"<{size // 4}I", sector(s))
    limit = len(fat)

    def read_chain(start: int, length: int | None = None) -> bytes:
        blob = b"".join(sector(s) for s in _chain(fat, start, limit))
        return blob if length is None else blob[:length]

    directory = read_chain(first_dir)
    entries = []
    for off in range(0, len(directory) - 127, 128):
        n = struct.unpack_from("<H", directory, off + 0x40)[0]
        kind = directory[off + 0x42]
        start, length = struct.unpack_from("<IQ", directory, off + 0x74)
        if major_v3(data):
            length &= 0xFFFFFFFF
        label = directory[off:off + max(n - 2, 0)].decode("utf-16-le", "replace")
        entries.append((label, kind, start, length))
    if not entries:
        raise _Bad("no directory")
    found = next((e for e in entries if e[1] == 2 and e[0].lower() == name.lower()), None)
    if found is None:
        return None
    _, _, start, length = found
    if length >= cutoff:
        return read_chain(start, length)
    root = entries[0]
    ministream = read_chain(root[2], root[3])
    minifat: list[int] = []
    if n_mini:
        raw = read_chain(first_mini)
        minifat = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    blob = b"".join(ministream[s * mini_size:(s + 1) * mini_size] for s in _chain(minifat, start, len(minifat)))
    return blob[:length]


def major_v3(data: bytes) -> bool:
    return struct.unpack_from("<H", data, 0x1A)[0] == 3


# -- BIFF8 records -------------------------------------------------------------------------------------------

def _records(book: bytes, pos: int):
    """(type, payload, CONTINUE payloads) from ``pos`` to the end of the stream."""
    n = len(book)
    while pos + 4 <= n:
        kind, length = struct.unpack_from("<HH", book, pos)
        payload = book[pos + 4:pos + 4 + length]
        pos += 4 + length
        conts = []
        while pos + 4 <= n and struct.unpack_from("<H", book, pos)[0] == 0x003C:
            clen = struct.unpack_from("<H", book, pos + 2)[0]
            conts.append(book[pos + 4:pos + 4 + clen])
            pos += 4 + clen
        yield kind, payload, conts


class _Reader:
    """Reads unicode strings across a record and its CONTINUE records."""

    def __init__(self, blocks: list[bytes]):
        self.blocks = blocks
        self.b = 0
        self.pos = 0

    def _need(self, n: int) -> None:
        if self.pos + n > len(self.blocks[self.b]):
            if self.pos == len(self.blocks[self.b]) and self.b + 1 < len(self.blocks):
                self.b += 1
                self.pos = 0
            if self.pos + n > len(self.blocks[self.b]):
                raise _Bad("a string runs past the end of its record")

    def take(self, n: int) -> bytes:
        self._need(n)
        out = self.blocks[self.b][self.pos:self.pos + n]
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        while n > 0:
            if self.pos == len(self.blocks[self.b]):
                if self.b + 1 >= len(self.blocks):
                    raise _Bad("data runs past the end of its record")
                self.b, self.pos = self.b + 1, 0
            step = min(n, len(self.blocks[self.b]) - self.pos)
            self.pos += step
            n -= step

    def string(self, long_count: bool = True) -> str:
        cch = struct.unpack("<H", self.take(2))[0] if long_count else self.take(1)[0]
        flags = self.take(1)[0]
        runs = struct.unpack("<H", self.take(2))[0] if flags & 0x08 else 0
        ext = struct.unpack("<I", self.take(4))[0] if flags & 0x04 else 0
        wide = flags & 0x01
        chars: list[str] = []
        left = cch
        while left:
            if self.pos == len(self.blocks[self.b]):
                # the characters continue in the next record, which starts with a new width flag
                if self.b + 1 >= len(self.blocks):
                    raise _Bad("a string runs past the end of its record")
                self.b, self.pos = self.b + 1, 0
                wide = self.take(1)[0] & 0x01
            width = 2 if wide else 1
            room = (len(self.blocks[self.b]) - self.pos) // width
            n = min(left, room)
            if n == 0:
                raise _Bad("a broken string")
            raw = self.take(n * width)
            chars.append(raw.decode("utf-16-le" if wide else "latin-1"))
            left -= n
        self.skip(4 * runs + ext)
        return "".join(chars)


def _rk(v: int) -> float:
    if v & 0x02:
        x = float(struct.unpack("<i", struct.pack("<I", v))[0] >> 2)
    else:
        x = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return x / 100.0 if v & 0x01 else x


def _globals(book: bytes) -> tuple[list[tuple[str, int]], list[str]]:
    sheets, sst = [], []
    first = True
    for kind, payload, conts in _records(book, 0):
        if first:
            if kind != 0x0809 or len(payload) < 4:
                raise _Bad("the workbook stream does not start with a BOF record")
            version = struct.unpack_from("<H", payload, 0)[0]
            if version != 0x0600:
                raise _Bad("only BIFF8 workbooks (Excel 97 to 2003) are supported; this one is older")
            first = False
            continue
        if kind == 0x002F:
            raise MatlabError("The .xls workbook is password protected and cannot be read.")
        if kind == 0x0085:
            pos, _vis, sheet_type = struct.unpack_from("<IBB", payload, 0)
            name = _Reader([payload[6:]]).string(long_count=False)
            if sheet_type == 0:
                sheets.append((name, pos))
        elif kind == 0x00FC:
            r = _Reader([payload] + conts)
            r.skip(4)
            unique = struct.unpack("<I", r.take(4))[0]
            for _ in range(unique):
                sst.append(r.string())
        elif kind == 0x000A:
            break
    return sheets, sst


def _sheet(book: bytes, pos: int, sst: list[str]) -> list[list]:
    cells: dict[tuple[int, int], Any] = {}
    pending: tuple[int, int] | None = None
    started = False
    for kind, payload, conts in _records(book, pos):
        if not started:
            if kind != 0x0809:
                raise _Bad("a worksheet does not start with a BOF record")
            started = True
            continue
        if kind == 0x000A:
            break
        if kind == 0x0203:  # NUMBER
            r, c, _, x = struct.unpack_from("<HHHd", payload, 0)
            cells[r, c] = x
        elif kind == 0x027E:  # RK
            r, c, _, v = struct.unpack_from("<HHHI", payload, 0)
            cells[r, c] = _rk(v)
        elif kind == 0x00BD:  # MULRK
            r, c0 = struct.unpack_from("<HH", payload, 0)
            for k in range((len(payload) - 6) // 6):
                cells[r, c0 + k] = _rk(struct.unpack_from("<I", payload, 4 + 6 * k + 2)[0])
        elif kind == 0x00FD:  # LABELSST
            r, c, _, i = struct.unpack_from("<HHHI", payload, 0)
            cells[r, c] = sst[i] if i < len(sst) else ""
        elif kind in (0x0204, 0x00D6):  # LABEL, RSTRING
            r, c = struct.unpack_from("<HH", payload, 0)
            cells[r, c] = _Reader([payload[6:]] + conts).string()
        elif kind == 0x0205:  # BOOLERR
            r, c, _, v, is_err = struct.unpack_from("<HHHBB", payload, 0)
            cells[r, c] = _ERRORS.get(v, "#ERROR") if is_err else bool(v)
        elif kind == 0x0006:  # FORMULA: the cached result
            r, c = struct.unpack_from("<HH", payload, 0)
            result = payload[6:14]
            if result[6:8] != b"\xff\xff":
                cells[r, c] = struct.unpack("<d", result)[0]
            elif result[0] == 0:
                pending = (r, c)  # the text follows in a STRING record
            elif result[0] == 1:
                cells[r, c] = bool(result[2])
            elif result[0] == 2:
                cells[r, c] = _ERRORS.get(result[2], "#ERROR")
            elif result[0] == 3:
                cells[r, c] = ""
        elif kind == 0x0207 and pending is not None:  # STRING after a formula
            cells[pending] = _Reader([payload] + conts).string()
            pending = None
    if not cells:
        return []
    n_rows = max(r for r, _ in cells) + 1
    rows: list[list] = [[] for _ in range(n_rows)]
    for (r, c), v in sorted(cells.items()):
        line = rows[r]
        line += [None] * (c + 1 - len(line))
        line[c] = v
    return rows


def _xlrd_read(xlrd, fname: str, sheet: Any, pick) -> tuple[list[list], list[str]]:
    try:
        wb = xlrd.open_workbook(fname, on_demand=True)
    except Exception as e:  # noqa: BLE001 - xlrd raises several unrelated error types
        raise MatlabError(f"'{fname}' is not an Excel 97-2003 workbook (.xls) that can be read ({e}).") from None
    names = list(wb.sheet_names())
    ws = wb.sheet_by_index(pick(names, sheet, fname))
    rows = []
    for r in range(ws.nrows):
        out: list = []
        for cell in ws.row(r):
            t, v = cell.ctype, cell.value
            if t in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                out.append(None)
            elif t == xlrd.XL_CELL_BOOLEAN:
                out.append(bool(v))
            elif t == xlrd.XL_CELL_ERROR:
                out.append(_ERRORS.get(v, "#ERROR"))
            elif t == xlrd.XL_CELL_TEXT:
                out.append(str(v))
            else:
                out.append(float(v))
        rows.append(out)
    wb.release_resources()
    return rows, names


def read(fname: str, sheet: Any, pick) -> tuple[list[list], list[str]]:
    """Rows of one sheet (picked by ``pick(names, sheet, fname)``) and the names of all sheets."""
    try:
        import xlrd
    except ImportError:
        xlrd = None
    if xlrd is not None and getattr(xlrd, "XL_CELL_BOOLEAN", None) is not None:
        return _xlrd_read(xlrd, fname, sheet, pick)
    with open(fname, "rb") as fh:
        data = fh.read()
    try:
        book = _stream(data, "Workbook")
        if book is None:
            if _stream(data, "Book") is not None:
                raise _Bad("only BIFF8 workbooks (Excel 97 to 2003) are supported; this one is older")
            raise _Bad("there is no Workbook stream")
        sheets, sst = _globals(book)
        names = [n for n, _ in sheets]
        if not sheets:
            raise _Bad("the workbook has no worksheets")
        k = pick(names, sheet, fname)
        return _sheet(book, sheets[k][1], sst), names
    except (_Bad, struct.error, IndexError, UnicodeDecodeError) as e:
        raise MatlabError(f"The contents of '{fname}' are not supported: it is not an Excel 97-2003 (BIFF8) "
                          f"workbook that can be read ({e}).") from None
