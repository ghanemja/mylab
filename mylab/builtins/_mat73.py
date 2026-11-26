"""Version 7.3 MAT-files: HDF5 files laid out the way MATLAB does, through the optional h5py package.

Layout (public knowledge about the format): every variable is a dataset or group carrying a
``MATLAB_class`` attribute. Arrays are stored with their dimensions reversed. Char data is uint16
and logical data uint8, both marked with ``MATLAB_int_decode``. An empty array stores its size as
data and has ``MATLAB_empty``. Complex numbers are a compound type with ``real`` and ``imag``.
A cell is a dataset of object references into the group ``#refs#``. A scalar struct is a group
with one member per field; a struct array stores each field as a dataset of references. A sparse
matrix is a group with ``data``, ``ir`` and ``jc`` and the attribute ``MATLAB_sparse``.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def h5py_module(what: str = "Version 7.3 MAT-files"):
    try:
        import h5py
    except ImportError:
        raise MatlabError(f"{what} are HDF5 files and need the optional Python package 'h5py', which is not "
                          "installed. Install it with 'pip install h5py'.", "mylab:hdf5:notInstalled") from None
    return h5py


def is_hdf5(head: bytes) -> bool:
    return head[:8] == HDF5_SIGNATURE or head[512:520] == HDF5_SIGNATURE


_CLASS_OF = {"float64": "double", "float32": "single", "complex128": "double", "complex64": "single"}


def _class_of(a: np.ndarray) -> str:
    if a.dtype == bool:
        return "logical"
    if a.dtype.kind == "U":
        return "char"
    return _CLASS_OF.get(a.dtype.name, a.dtype.name)


class _Writer:
    def __init__(self, h5, f):
        self.h5 = h5
        self.f = f
        self.count = 0
        if "#refs#" in f:
            self.count = len(f["#refs#"])

    def _ref(self, v: Any):
        refs = self.f.require_group("#refs#")
        while True:
            name = _ref_name(self.count)
            self.count += 1
            if name not in refs:
                break
        self.write(refs, name, v)
        return refs[name].ref

    def _attrs(self, obj, cls: str, **more) -> None:
        obj.attrs["MATLAB_class"] = np.bytes_(cls)
        for k, v in more.items():
            obj.attrs[k] = v

    def _empty(self, parent, name: str, cls: str, shape: tuple, **more) -> None:
        ds = parent.create_dataset(name, data=np.array(shape, dtype=np.uint64))
        self._attrs(ds, cls, MATLAB_empty=np.uint8(1), **more)

    def _refs_dataset(self, parent, name: str, items: list, shape: tuple):
        data = np.empty(len(items), dtype=self.h5.ref_dtype)
        for k, x in enumerate(items):
            data[k] = self._ref(x)
        # items come in column-major order: reversed dimensions in C order
        return parent.create_dataset(name, data=data.reshape(shape[::-1]), dtype=self.h5.ref_dtype)

    def write(self, parent, name: str, v: Any) -> None:
        from .sparse import SparseMatrix
        if isinstance(v, SparseMatrix):
            m = v.m
            g = parent.create_group(name)
            cls = "logical" if m.dtype == bool else "double"
            self._attrs(g, cls, MATLAB_sparse=np.uint64(m.shape[0]))
            if m.nnz:
                data = m.data.astype(np.uint8) if m.dtype == bool else m.data
                if data.dtype.kind == "c":
                    data = _split_complex(data)
                g.create_dataset("data", data=data)
                g.create_dataset("ir", data=m.indices.astype(np.uint64))
            g.create_dataset("jc", data=m.indptr.astype(np.uint64))
            return
        if isinstance(v, Cell):
            if v.data.size == 0:
                self._empty(parent, name, "cell", v.data.shape)
                return
            ds = self._refs_dataset(parent, name, list(v.data.flatten(order="F")), v.data.shape)
            self._attrs(ds, "cell")
            return
        if isinstance(v, Struct):
            fields = list(v.fields)
            if v.data.size == 0 or not fields:
                self._empty(parent, name, "struct", v.data.shape)
                if fields:
                    parent[name].attrs.create("MATLAB_fields", _field_names(self.h5, fields),
                                              dtype=self.h5.vlen_dtype(np.dtype("S1")))
                return
            g = parent.create_group(name)
            self._attrs(g, "struct")
            g.attrs.create("MATLAB_fields", _field_names(self.h5, fields), dtype=self.h5.vlen_dtype(np.dtype("S1")))
            if v.data.size == 1:
                for f in fields:
                    self.write(g, f, v.data.flat[0][f])
            else:
                for f in fields:
                    self._refs_dataset(g, f, [d[f] for d in v.data.flatten(order="F")], v.data.shape)
            return
        if not isinstance(v, np.ndarray):
            raise MatlabError(f"Values of type {type(v).__name__} cannot be stored in a MAT-file.")
        cls = _class_of(v)
        if v.size == 0:
            more = {"MATLAB_int_decode": np.int32(2 if cls == "char" else 1)} if cls in ("char", "logical") else {}
            self._empty(parent, name, cls, v.shape, **more)
            return
        t = np.ascontiguousarray(v.transpose())
        if cls == "char":
            codes = np.ascontiguousarray(t.astype("<U1")).view("<u4").reshape(t.shape)
            if int(codes.max()) > 0xFFFF:
                raise MatlabError("Characters outside the Basic Multilingual Plane cannot be stored in a char "
                                  "array of a MAT-file.")
            ds = parent.create_dataset(name, data=codes.astype(np.uint16))
            self._attrs(ds, cls, MATLAB_int_decode=np.int32(2))
        elif cls == "logical":
            ds = parent.create_dataset(name, data=t.astype(np.uint8))
            self._attrs(ds, cls, MATLAB_int_decode=np.int32(1))
        elif t.dtype.kind == "c":
            ds = parent.create_dataset(name, data=_split_complex(t))
            self._attrs(ds, cls)
        else:
            big = t.nbytes > 1 << 16
            ds = parent.create_dataset(name, data=t, **({"compression": "gzip", "compression_opts": 3,
                                                         "chunks": True} if big else {}))
            self._attrs(ds, cls)


def _split_complex(a: np.ndarray) -> np.ndarray:
    part = np.float32 if a.dtype == np.complex64 else np.float64
    out = np.empty(a.shape, dtype=[("real", part), ("imag", part)])
    out["real"] = a.real
    out["imag"] = a.imag
    return out


def _field_names(h5, fields: list[str]) -> np.ndarray:
    out = np.empty(len(fields), dtype=object)
    for k, f in enumerate(fields):
        out[k] = np.frombuffer(f.encode("ascii", errors="replace"), dtype="S1")
    return out


def _ref_name(k: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, len(letters))
        s = letters[r] + s
    return s


def _userblock() -> bytes:
    import time
    text = ("MATLAB 7.3 MAT-file, Platform: mylab, Created on: " + time.strftime("%a %b %d %H:%M:%S %Y")
            + " HDF5 schema 1.00 .")
    head = text.encode("ascii")[:116].ljust(116, b" ") + b"\0" * 8 + struct.pack("<H", 0x0200) + b"IM"
    return head.ljust(512, b"\0")


def write(path: str, variables: dict, append: bool) -> None:
    """Write (or add to) a version 7.3 file. ``variables`` maps names to standard values."""
    import os
    h5 = h5py_module()
    exists = append and os.path.isfile(path)
    try:
        if exists:
            f = h5.File(path, "r+")
        else:
            f = h5.File(path, "w", userblock_size=512, libver="earliest")
    except OSError as e:
        raise MatlabError(f"Unable to write the file '{path}': {e}") from None
    try:
        w = _Writer(h5, f)
        for name, v in variables.items():
            if name in f:
                del f[name]
            w.write(f, name, v)
    finally:
        f.close()
    if not exists:
        with open(path, "r+b") as fh:
            fh.write(_userblock())


class _Reader:
    def __init__(self, h5, f, warn):
        self.h5 = h5
        self.f = f
        self.warn = warn

    def read(self, obj: Any) -> Any:
        h5 = self.h5
        cls = obj.attrs.get("MATLAB_class", b"")
        cls = cls.decode("ascii", errors="replace") if isinstance(cls, bytes) else str(cls)
        if isinstance(obj, h5.Group):
            if "MATLAB_sparse" in obj.attrs:
                return self._sparse(obj, cls)
            return self._struct(obj)
        if "MATLAB_empty" in obj.attrs and int(np.asarray(obj.attrs["MATLAB_empty"]).flat[0]):
            shape = tuple(int(x) for x in np.asarray(obj[()]).flatten())
            if len(shape) < 2:
                shape = (0, 0)
            if cls == "cell":
                return Cell(np.empty(shape, dtype=object))
            if cls == "struct":
                return Struct(np.empty(shape, dtype=object), self._fields(obj) or [])
            if cls == "char":
                return np.zeros(shape, dtype="<U1")
            if cls == "logical":
                return np.zeros(shape, dtype=bool)
            return np.zeros(shape, dtype=_dtype_for(cls))
        if cls == "canonical empty":
            return np.zeros((0, 0))
        data = obj[()]
        if obj.dtype == h5.ref_dtype or (obj.dtype.kind == "O" and h5.check_ref_dtype(obj.dtype)):
            arr = np.asarray(data, dtype=object)
            shape = arr.shape[::-1] if arr.ndim >= 2 else (1, arr.size) if arr.ndim == 1 else (1, 1)
            flat = arr.reshape(-1)  # C order of the reversed shape = column-major order
            out = np.empty(len(flat), dtype=object)
            for k, r in enumerate(flat):
                out[k] = self.read(self.f[r]) if r else np.zeros((0, 0))
            return Cell(out.reshape(shape, order="F"))
        if cls in ("function_handle",) or "MATLAB_object_decode" in obj.attrs:
            self.warn(f"a '{cls}' value stored by MATLAB in a version 7.3 file cannot be read; it was loaded as "
                      "an empty array.")
            return np.zeros((0, 0))
        arr = np.asarray(data)
        if arr.dtype.names and set(arr.dtype.names) >= {"real", "imag"}:
            arr = arr["real"] + 1j * arr["imag"]
        arr = arr.transpose() if arr.ndim >= 2 else arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(1, 1)
        if cls == "char":
            codes = np.ascontiguousarray(arr.astype("<u4"))
            return codes.view("<U1").reshape(arr.shape)
        if cls == "logical":
            return arr != 0
        if cls and cls in _DTYPES and arr.dtype.kind != "c":
            return np.ascontiguousarray(arr.astype(_DTYPES[cls]))
        return np.ascontiguousarray(arr)

    def _fields(self, obj) -> list[str] | None:
        if "MATLAB_fields" not in obj.attrs:
            return None
        out = []
        for item in np.atleast_1d(obj.attrs["MATLAB_fields"]):
            if isinstance(item, (bytes, str)):
                out.append(item.decode() if isinstance(item, bytes) else item)
            else:
                out.append(b"".join(np.asarray(item).reshape(-1).tolist()).decode("ascii", errors="replace"))
        return out

    def _struct(self, g) -> Struct:
        h5 = self.h5
        fields = self._fields(g) or [k for k in g.keys()]
        fields = [f for f in fields if f in g]
        is_array = bool(fields) and all(
            isinstance(g[f], h5.Dataset) and "MATLAB_class" not in g[f].attrs
            and (g[f].dtype == h5.ref_dtype or (g[f].dtype.kind == "O" and h5.check_ref_dtype(g[f].dtype)))
            for f in fields)
        if not is_array:
            data = np.empty((1, 1), dtype=object)
            data[0, 0] = {f: self.read(g[f]) for f in fields}
            return Struct(data, fields)
        cols = {f: self.read(g[f]) for f in fields}  # each a Cell of the struct array's shape
        shape = cols[fields[0]].data.shape
        data = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            data[idx] = {f: cols[f].data[idx] for f in fields}
        return Struct(data, fields)

    def _sparse(self, g, cls: str):
        import scipy.sparse as sp

        from .sparse import SparseMatrix
        rows = int(np.asarray(g.attrs["MATLAB_sparse"]).flat[0])
        jc = np.asarray(g["jc"][()]).astype(np.int64).reshape(-1)
        cols = jc.size - 1
        if "data" in g:
            data = np.asarray(g["data"][()]).reshape(-1)
            if data.dtype.names:
                data = data["real"] + 1j * data["imag"]
            ir = np.asarray(g["ir"][()]).astype(np.int64).reshape(-1)
        else:
            data, ir = np.zeros(0), np.zeros(0, dtype=np.int64)
        data = (data != 0) if cls == "logical" else data.astype(np.complex128 if data.dtype.kind == "c" else float)
        return SparseMatrix(sp.csc_matrix((data, ir, jc), shape=(rows, cols)))


_DTYPES = {"double": np.float64, "single": np.float32, "int8": np.int8, "uint8": np.uint8, "int16": np.int16,
           "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32, "int64": np.int64, "uint64": np.uint64}


def _dtype_for(cls: str):
    return _DTYPES.get(cls, np.float64)


def names(path: str) -> list[str]:
    h5 = h5py_module()
    with h5.File(path, "r") as f:
        return [k for k in f.keys() if not k.startswith("#")]


def read(path: str, wanted, warn) -> dict:
    """Variables of a version 7.3 file as standard values. ``wanted(name)`` selects them."""
    h5 = h5py_module()
    try:
        f = h5.File(path, "r")
    except OSError as e:
        raise MatlabError(f"Unable to read the file '{path}': {e}") from None
    out = {}
    with f:
        r = _Reader(h5, f, warn)
        for name in f.keys():
            if name.startswith("#") or not wanted(name):
                continue
            out[name] = r.read(f[name])
    return out
