"""MAT-file containers: the Level 5 binary format used by MATLAB's -v6 and -v7 files.

This module only knows the container. It reads a file into mylab values (numeric, logical and
char arrays, sparse matrices, cells, structs) and writes those same types back. Objects that
have no standard MAT representation are turned into plain structs first by ``_matcodec``.

Layout of a Level 5 file, as documented publicly: a 128-byte header (text, the offset of the
"subsystem" data, a version and the two bytes ``MI`` whose order gives the byte order), then a
sequence of tagged data elements. A variable is an ``miMATRIX`` element, optionally wrapped in
an ``miCOMPRESSED`` (zlib) element.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, Struct

# data types of an element
MI_INT8, MI_UINT8, MI_INT16, MI_UINT16, MI_INT32, MI_UINT32, MI_SINGLE = 1, 2, 3, 4, 5, 6, 7
MI_DOUBLE, MI_INT64, MI_UINT64, MI_MATRIX, MI_COMPRESSED, MI_UTF8, MI_UTF16, MI_UTF32 = 9, 12, 13, 14, 15, 16, 17, 18
_MI_CODE = {MI_INT8: "i1", MI_UINT8: "u1", MI_INT16: "i2", MI_UINT16: "u2", MI_INT32: "i4", MI_UINT32: "u4",
            MI_SINGLE: "f4", MI_DOUBLE: "f8", MI_INT64: "i8", MI_UINT64: "u8", MI_UTF8: "u1", MI_UTF16: "u2",
            MI_UTF32: "u4"}
_CODE_MI = {"i1": MI_INT8, "u1": MI_UINT8, "i2": MI_INT16, "u2": MI_UINT16, "i4": MI_INT32, "u4": MI_UINT32,
            "f4": MI_SINGLE, "f8": MI_DOUBLE, "i8": MI_INT64, "u8": MI_UINT64}

# array classes
MX_CELL, MX_STRUCT, MX_OBJECT, MX_CHAR, MX_SPARSE, MX_DOUBLE, MX_SINGLE = 1, 2, 3, 4, 5, 6, 7
MX_INT8, MX_UINT8, MX_INT16, MX_UINT16, MX_INT32, MX_UINT32, MX_INT64, MX_UINT64 = 8, 9, 10, 11, 12, 13, 14, 15
MX_FUNCTION, MX_OPAQUE = 16, 17
_MX_DTYPE = {MX_DOUBLE: np.float64, MX_SINGLE: np.float32, MX_INT8: np.int8, MX_UINT8: np.uint8,
             MX_INT16: np.int16, MX_UINT16: np.uint16, MX_INT32: np.int32, MX_UINT32: np.uint32,
             MX_INT64: np.int64, MX_UINT64: np.uint64}
_DTYPE_MX = {np.dtype(v): k for k, v in _MX_DTYPE.items()}
MX_NAMES = {MX_CELL: "cell", MX_STRUCT: "struct", MX_OBJECT: "object", MX_CHAR: "char", MX_SPARSE: "double",
            MX_DOUBLE: "double", MX_SINGLE: "single", MX_INT8: "int8", MX_UINT8: "uint8", MX_INT16: "int16",
            MX_UINT16: "uint16", MX_INT32: "int32", MX_UINT32: "uint32", MX_INT64: "int64",
            MX_UINT64: "uint64", MX_FUNCTION: "function_handle", MX_OPAQUE: "opaque"}

FLAG_COMPLEX, FLAG_GLOBAL, FLAG_LOGICAL = 0x08, 0x04, 0x02


class Opaque:
    """An object stored through MATLAB's object system (class 17): resolved later by the codec."""

    def __init__(self, type_system: str, class_name: str, payload: Any):
        self.type_system = type_system
        self.class_name = class_name
        self.payload = payload


class OldObject:
    """An old-style object (class 3): a class name plus a struct of fields."""

    def __init__(self, class_name: str, fields: Struct):
        self.class_name = class_name
        self.fields = fields


class StoredFunction:
    """A function handle as MATLAB stores it (class 16): a struct describing the function."""

    def __init__(self, info: Any):
        self.info = info


class CorruptFile(MatlabError):
    pass


def _bad(what: str) -> CorruptFile:
    return CorruptFile(f"The MAT-file is damaged or not a Level 5 MAT-file ({what}).")


# -- reading ---------------------------------------------------------------------------------------

class Reader:
    """Decodes data elements from a bytes buffer in the given byte order ('<' or '>')."""

    def __init__(self, buf: bytes, endian: str, strict: bool = True):
        self.buf = buf
        self.e = endian
        self.strict = strict  # False when only the first bytes of an element are in the buffer
        self._u4 = struct.Struct(endian + "I")

    def tag(self, pos: int, container: bool = False) -> tuple[int, int, int, int]:
        """(data type, byte count, start of the data, position after the padded element)."""
        if pos + 8 > len(self.buf):
            raise _bad("unexpected end of data")
        first = self._u4.unpack_from(self.buf, pos)[0]
        small = first >> 16
        if small:  # small element: type and count share the first four bytes, data in the next four
            if small > 4:
                raise _bad("bad small data element")
            return first & 0xFFFF, small, pos + 4, pos + 8
        n = self._u4.unpack_from(self.buf, pos + 4)[0]
        if self.strict and not container and pos + 8 + n > len(self.buf):
            raise _bad("an element is longer than the file")
        return first, n, pos + 8, pos + 8 + ((n + 7) & ~7)

    def raw(self, pos: int) -> tuple[int, bytes, int]:
        mdtype, n, start, nxt = self.tag(pos)
        return mdtype, self.buf[start:start + n], nxt

    def numbers(self, pos: int) -> tuple[np.ndarray, int, int]:
        """(values, data type, next position) of a numeric element."""
        mdtype, n, start, nxt = self.tag(pos)
        code = _MI_CODE.get(mdtype)
        if code is None:
            raise _bad(f"unknown data type {mdtype}")
        dt = np.dtype(self.e + code)
        return np.frombuffer(self.buf, dtype=dt, count=n // dt.itemsize, offset=start), mdtype, nxt

    def text(self, pos: int) -> tuple[str, int]:
        mdtype, data, nxt = self.raw(pos)
        return data.split(b"\0", 1)[0].decode("utf-8", errors="replace"), nxt

    # one array -----------------------------------------------------------------------------------
    def header(self, pos: int) -> dict:
        """Class, flags, dimensions and name of the miMATRIX element at ``pos`` (cheap)."""
        mdtype, n, start, nxt = self.tag(pos, container=True)
        if mdtype != MI_MATRIX:
            raise _bad(f"expected an array, found data type {mdtype}")
        info = {"start": start, "next": nxt, "empty": n == 0, "name": "", "dims": (0, 0),
                "class": MX_DOUBLE, "flags": 0, "nzmax": 0, "body": start}
        if n == 0:
            return info
        flags, _, p = self.numbers(start)
        if flags.size < 2:
            raise _bad("bad array flags")
        word = int(flags[0])
        info["class"] = word & 0xFF
        info["flags"] = (word >> 8) & 0xFF
        info["nzmax"] = int(flags[1])
        if info["class"] == MX_OPAQUE:
            info["name"], p = self.text(p)
            info["dims"] = (1, 1)
        else:
            dims, _, p = self.numbers(p)
            info["dims"] = tuple(int(d) for d in dims)
            info["name"], p = self.text(p)
        info["body"] = p
        return info

    def matrix(self, pos: int) -> tuple[str, Any, dict, int]:
        """(name, value, header, position after the element) of the miMATRIX element at ``pos``.

        The end position comes from the data actually read, not from the length the element
        declares: some writers declare a wrong length for short char arrays.
        """
        h = self.header(pos)
        if h["empty"]:
            return "", np.zeros((0, 0)), h, h["start"]
        cls, dims, p = h["class"], h["dims"], h["body"]
        if len(dims) < 2:
            dims = tuple(dims) + (1,) * (2 - len(dims))
        count = int(np.prod(dims, dtype=np.int64))
        is_complex = bool(h["flags"] & FLAG_COMPLEX)
        is_logical = bool(h["flags"] & FLAG_LOGICAL)
        if cls in _MX_DTYPE:
            try:
                value, p = self._numeric(p, cls, dims, is_complex, is_logical)
            except CorruptFile:
                if not is_logical:
                    raise
                # Octave writes a sparse logical matrix under the class of a full one
                value, p = self._sparse(p, dims, False, True)
        elif cls == MX_CHAR:
            value, p = self._char(p, dims, count)
        elif cls == MX_SPARSE:
            value, p = self._sparse(p, dims, is_complex, is_logical)
        elif cls == MX_CELL:
            data = np.empty(count, dtype=object)
            for k in range(count):
                _, data[k], _, p = self.matrix(p)
            value = Cell(data.reshape(dims, order="F"))
        elif cls in (MX_STRUCT, MX_OBJECT):
            class_name = ""
            if cls == MX_OBJECT:
                class_name, p = self.text(p)
            value, p = self._struct(p, dims, count)
            if cls == MX_OBJECT:
                value = OldObject(class_name, value)
        elif cls == MX_FUNCTION:
            _, info, _, p = self.matrix(p)
            value = StoredFunction(info)
        elif cls == MX_OPAQUE:
            type_system, p = self.text(p)
            class_name, p = self.text(p)
            _, payload, _, p = self.matrix(p)
            value = Opaque(type_system, class_name, payload)
        else:
            raise _bad(f"unknown array class {cls}")
        return h["name"], value, h, p

    def _numeric(self, p: int, cls: int, dims: tuple, is_complex: bool, is_logical: bool):
        dtype = _MX_DTYPE[cls]
        re, _, p = self.numbers(p)
        if is_complex:
            im, _, p = self.numbers(p)
            ctype = np.complex64 if cls == MX_SINGLE else np.complex128
            arr = re.astype(ctype)
            arr.imag = im
        elif is_logical:
            arr = re != 0
        else:
            arr = re.astype(dtype)  # data may be stored in a smaller type than the array's class
        try:
            return arr.reshape(dims, order="F"), p
        except ValueError:
            raise _bad("the data does not match the array size") from None

    def _char(self, p: int, dims: tuple, count: int):
        mdtype, data, p = self.raw(p)
        text = None
        if mdtype == MI_UTF8:
            text = data.decode("utf-8", errors="replace")
        elif mdtype == MI_UTF16:
            text = data.decode("utf-16-le" if self.e == "<" else "utf-16-be", errors="replace")
        elif mdtype == MI_UTF32:
            text = data.decode("utf-32-le" if self.e == "<" else "utf-32-be", errors="replace")
        else:
            code = _MI_CODE.get(mdtype)
            if code is None:
                raise _bad(f"unknown character data type {mdtype}")
            dt = np.dtype(self.e + code)
            units = np.frombuffer(data, dtype=dt, count=len(data) // dt.itemsize).astype(np.uint32)
        if text is not None:
            units = np.array([ord(c) for c in text], dtype=np.uint32)
        if units.size != count:
            fixed = np.full(count, 32, dtype=np.uint32)
            fixed[:min(count, units.size)] = units[:count]
            units = fixed
        units = np.where(((units >= 0xD800) & (units <= 0xDFFF)) | (units > 0x10FFFF), 0xFFFD, units)
        chars = units.astype("<u4").view("<U1") if count else np.zeros(0, dtype="<U1")
        return chars.reshape(dims, order="F"), p

    def _sparse(self, p: int, dims: tuple, is_complex: bool, is_logical: bool):
        import scipy.sparse as sp

        from .sparse import SparseMatrix
        ir, _, p = self.numbers(p)
        jc, _, p = self.numbers(p)
        nnz = int(jc[-1]) if jc.size else 0
        if is_logical:
            # MATLAB stores one byte per element here, whatever type the tag names; Octave stores doubles
            mdtype, raw, p = self.raw(p)
            dt = np.dtype(self.e + _MI_CODE.get(mdtype, "u1"))
            if dt.itemsize > 1 and len(raw) >= nnz * dt.itemsize:
                data = np.frombuffer(raw, dtype=dt, count=nnz) != 0
            else:
                data = np.frombuffer(raw, dtype=np.uint8)
                data = (data[:nnz] != 0) if data.size >= nnz else np.ones(nnz, dtype=bool)
        else:
            re, _, p = self.numbers(p)
            if is_complex:
                im, _, p = self.numbers(p)
                data = re[:nnz].astype(np.complex128)
                data.imag = im[:nnz]
            else:
                data = re[:nnz].astype(np.float64)
        try:
            m = sp.csc_matrix((data, ir[:nnz].astype(np.int32), jc.astype(np.int32)), shape=tuple(dims[:2]))
        except ValueError:
            raise _bad("bad sparse matrix") from None
        return SparseMatrix(m), p

    def _struct(self, p: int, dims: tuple, count: int) -> tuple[Struct, int]:
        width, _, p = self.numbers(p)
        _, names_raw, p = self.raw(p)
        w = int(width[0]) if width.size else 32
        fields: list[str] = []
        for k in range(len(names_raw) // w if w > 0 else 0):
            name = names_raw[k * w:(k + 1) * w].split(b"\0", 1)[0].decode("utf-8", errors="replace")
            while name in fields or name == "":  # duplicate names appear in some hand-made files
                name += "_"
            fields.append(name)
        data = np.empty(count, dtype=object)
        for k in range(count):
            d = {}
            for f in fields:
                _, d[f], _, p = self.matrix(p)
            data[k] = d
        return Struct(data.reshape(dims, order="F"), fields), p


class MatFile5:
    """A Level 5 MAT-file held in memory: lists, reads and rewrites its variables."""

    def __init__(self, data: bytes):
        if len(data) < 128:
            raise _bad("the file is shorter than its header")
        mark = data[126:128]
        if mark == b"IM":
            self.endian = "<"
        elif mark == b"MI":
            self.endian = ">"
        else:
            raise _bad("no byte-order mark")
        self.data = data
        self.header_text = data[:116].decode("latin-1").rstrip(" \0")
        off = struct.unpack_from(self.endian + "Q", data, 116)[0]
        self.subsys_offset = off if 128 <= off < len(data) and data[116:124] != b" " * 8 else 0
        self.damage = ""  # why reading stopped before the end of the file, if it did
        self._elements: list[dict] | None = None
        self._subsystem: bytes | None = None

    @staticmethod
    def is_level5(head: bytes) -> bool:
        return len(head) >= 128 and head[126:128] in (b"IM", b"MI") and head[:4] != b"\0\0\0\0"

    def _plausible(self, pos: int) -> bool:
        if pos == len(self.data):
            return True
        if pos + 8 > len(self.data):
            return False
        first, n = struct.unpack_from(self.endian + "II", self.data, pos)
        return first in (MI_MATRIX, MI_COMPRESSED) and pos + 8 + n <= len(self.data) + 8

    def elements(self) -> list[dict]:
        """One entry per variable: name, class, flags, dims, and where its bytes are."""
        if self._elements is not None:
            return self._elements
        out = []
        top = Reader(self.data, self.endian)
        pos = 128
        keys = ("name", "dims", "class", "flags", "nzmax")
        while pos + 8 <= len(self.data):
            first, n = struct.unpack_from(self.endian + "II", self.data, pos)
            if first == 0 and n == 0 and not any(self.data[pos:]):
                break  # zero padding at the end of the file
            entry = {"pos": pos, "compressed": first == MI_COMPRESSED}
            if first not in (MI_COMPRESSED, MI_MATRIX) or pos + 8 + n > len(self.data) + 8:
                self.damage = f"unreadable data at byte {pos}"
                break
            if first == MI_COMPRESSED:
                start = pos + 8
                entry["end"] = start + n
                entry["zdata"] = (start, n)
                try:
                    head = zlib.decompressobj().decompress(self.data[start:start + n], 4096)
                    h = Reader(head, self.endian, strict=False).header(0)
                except (zlib.error, CorruptFile, struct.error, ValueError, IndexError):
                    self.damage = f"a compressed variable at byte {pos} cannot be read"
                    pos = start + n
                    continue
            else:
                entry["end"] = pos + 8 + ((n + 7) & ~7)
                try:
                    h = top.header(pos)
                    if not self._plausible(entry["end"]):
                        entry["end"] = top.matrix(pos)[3]  # the declared length is wrong: measure it
                        if not self._plausible(entry["end"]):
                            raise _bad("bad element length")
                except (CorruptFile, struct.error, ValueError, IndexError):
                    self.damage = f"unreadable data at byte {pos}"
                    break
            entry.update({k: h[k] for k in keys})
            entry["subsystem"] = bool(self.subsys_offset) and pos == self.subsys_offset
            out.append(entry)
            pos = entry["end"]
        self._elements = out
        return out

    def variables(self) -> list[dict]:
        """The named variables; when a name occurs twice, the later one wins (as when loading)."""
        last: dict[str, dict] = {}
        for e in self.elements():
            if not e["subsystem"] and e["name"]:
                last.pop(e["name"], None)
                last[e["name"]] = e
        return list(last.values())

    def _bytes_of(self, entry: dict) -> tuple[Reader, int]:
        if entry["compressed"]:
            start, n = entry["zdata"]
            try:
                raw = zlib.decompress(self.data[start:start + n])
            except zlib.error:
                # a damaged checksum at the end still leaves the data usable
                try:
                    raw = zlib.decompressobj().decompress(self.data[start:start + n])
                except zlib.error:
                    raise _bad(f"cannot decompress variable '{entry.get('name', '')}'") from None
            return Reader(raw, self.endian), 0
        return Reader(self.data, self.endian), entry["pos"]

    def read(self, entry: dict) -> Any:
        reader, pos = self._bytes_of(entry)
        try:
            return reader.matrix(pos)[1]
        except (struct.error, IndexError, ValueError) as e:
            raise _bad(f"cannot read variable '{entry.get('name', '')}': {e}") from None

    def subsystem(self) -> bytes | None:
        """The raw subsystem bytes (object data), if the file has any."""
        if self._subsystem is None:
            for e in self.elements():
                if e["subsystem"]:
                    v = self.read(e)
                    if isinstance(v, np.ndarray):
                        self._subsystem = v.astype(np.uint8).tobytes()
                    break
        return self._subsystem

    def raw_element(self, entry: dict) -> bytes:
        """The bytes of one top-level element, exactly as stored (for copying into a new file)."""
        return self.data[entry["pos"]:entry["end"]]


# -- writing ---------------------------------------------------------------------------------------

def _element(mdtype: int, data: bytes) -> bytes:
    n = len(data)
    if 0 < n <= 4:
        return struct.pack("<HH", mdtype, n) + data + b"\0" * (4 - n)
    return struct.pack("<II", mdtype, n) + data + b"\0" * (-n % 8)


def _smallest_int(a: np.ndarray) -> np.ndarray:
    """Whole-number doubles are stored in the smallest integer type that holds them, as MATLAB does."""
    if a.size == 0 or a.dtype != np.float64:
        return a
    if not np.all(np.isfinite(a)) or np.any(a != np.floor(a)):
        return a
    if np.any((a == 0) & np.signbit(a)):
        return a  # keep negative zero
    lo, hi = float(a.min()), float(a.max())
    for dt in (np.uint8, np.int16, np.uint16, np.int32):
        info = np.iinfo(dt)
        if lo >= info.min and hi <= info.max:
            return a.astype(dt)
    return a


def _numbers(a: np.ndarray, compact: bool = False) -> bytes:
    a = np.ascontiguousarray(a.reshape(-1, order="F"))
    if compact:
        a = _smallest_int(a)
    dt = a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype
    code = dt.kind + str(dt.itemsize)
    return _element(_CODE_MI[code], a.astype(dt, copy=False).tobytes())


def _array_head(cls: int, flags: int, dims: tuple, name: str, nzmax: int = 0) -> bytes:
    return (_element(MI_UINT32, struct.pack("<II", cls | (flags << 8), nzmax))
            + _element(MI_INT32, np.array(dims, dtype="<i4").tobytes())
            + _element(MI_INT8, name.encode("utf-8")))


def _wrap(body: bytes) -> bytes:
    return struct.pack("<II", MI_MATRIX, len(body)) + body


def encode_array(v: Any, name: str = "", is_global: bool = False) -> bytes:
    """One miMATRIX element for a standard value (arrays, sparse, Cell, Struct)."""
    from .sparse import SparseMatrix
    g = FLAG_GLOBAL if is_global else 0
    if isinstance(v, SparseMatrix):
        m = v.m
        is_logical = m.dtype == bool
        is_complex = m.dtype.kind == "c"
        flags = g | (FLAG_LOGICAL if is_logical else 0) | (FLAG_COMPLEX if is_complex else 0)
        nnz = int(m.nnz)
        body = _array_head(MX_SPARSE, flags, m.shape, name, max(nnz, 1))
        body += _element(MI_INT32, m.indices.astype("<i4").tobytes())
        body += _element(MI_INT32, m.indptr.astype("<i4").tobytes())
        if is_logical:
            body += _element(MI_UINT8, m.data.astype(np.uint8).tobytes())
        elif is_complex:
            body += _numbers(m.data.real.astype(np.float64)) + _numbers(m.data.imag.astype(np.float64))
        else:
            body += _numbers(m.data.astype(np.float64))
        return _wrap(body)
    if isinstance(v, Cell):
        body = _array_head(MX_CELL, g, v.data.shape, name)
        return _wrap(body + b"".join(encode_array(x) for x in v.data.flatten(order="F")))
    if isinstance(v, Struct):
        fields = list(v.fields)
        width = max([len(f.encode("utf-8")) for f in fields] + [31]) + 1
        if width > 64:
            long_name = next(f for f in fields if len(f.encode("utf-8")) >= 64)
            raise MatlabError(f"The field name '{long_name}' is longer than the 63 characters a MAT-file "
                              "allows.")
        width = 32 if width <= 32 else 64
        body = _array_head(MX_STRUCT, g, v.data.shape, name)
        body += struct.pack("<HHi", MI_INT32, 4, width)
        body += _element(MI_INT8, b"".join(f.encode("utf-8").ljust(width, b"\0") for f in fields))
        parts = []
        for d in v.data.flatten(order="F"):
            for f in fields:
                parts.append(encode_array(d[f]))
        return _wrap(body + b"".join(parts))
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            codes = np.zeros(v.shape, dtype=np.uint32)
            if v.size:
                flat = np.ascontiguousarray(v.astype("<U1")).view("<u4").reshape(v.shape)
                codes = flat
            if codes.size and int(codes.max()) > 0xFFFF:
                raise MatlabError("Characters outside the Basic Multilingual Plane cannot be stored in a char "
                                  "array of a MAT-file.")
            body = _array_head(MX_CHAR, g, v.shape, name)
            return _wrap(body + _element(MI_UINT16, codes.reshape(-1, order="F").astype("<u2").tobytes()))
        if v.dtype == bool:
            body = _array_head(MX_UINT8, g | FLAG_LOGICAL, v.shape, name)
            return _wrap(body + _element(MI_UINT8, v.reshape(-1, order="F").astype(np.uint8).tobytes()))
        if v.dtype.kind == "c":
            cls = MX_SINGLE if v.dtype == np.complex64 else MX_DOUBLE
            part = np.float32 if cls == MX_SINGLE else np.float64
            body = _array_head(cls, g | FLAG_COMPLEX, v.shape, name)
            return _wrap(body + _numbers(v.real.astype(part), cls == MX_DOUBLE)
                         + _numbers(v.imag.astype(part), cls == MX_DOUBLE))
        cls = _DTYPE_MX.get(np.dtype(v.dtype.kind + str(v.dtype.itemsize)))
        if cls is None:
            raise MatlabError(f"Arrays of type {v.dtype} cannot be stored in a MAT-file.")
        body = _array_head(cls, g, v.shape, name)
        return _wrap(body + _numbers(v, cls == MX_DOUBLE))
    if isinstance(v, FunctionHandle):  # the codec replaces handles before they get here
        raise MatlabError("Function handles must be encoded before they are written.")
    raise MatlabError(f"Values of type {type(v).__name__} cannot be stored in a MAT-file.")


def file_header(description: str | None = None) -> bytes:
    import time
    text = description or ("MATLAB 5.0 MAT-file, Platform: mylab, Created on: "
                           + time.strftime("%a %b %d %H:%M:%S %Y"))
    return text.encode("ascii", errors="replace")[:116].ljust(116, b" ") + b" " * 8 + struct.pack("<H", 0x0100) + b"IM"


def top_level(element: bytes, compress: bool) -> bytes:
    """A variable's element as it appears in the file (deflated for -v7)."""
    if not compress:
        return element
    z = zlib.compress(element)
    return struct.pack("<II", MI_COMPRESSED, len(z)) + z


