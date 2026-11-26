"""Version 7.3 MAT-files: HDF5 files laid out the way MATLAB does, through the h5py package.

Layout (public knowledge about the format): every variable is a dataset or group carrying a
``MATLAB_class`` attribute. Arrays are stored with their dimensions reversed. Char data is uint16
and logical data uint8, both marked with ``MATLAB_int_decode``. An empty array stores its size as
data and has ``MATLAB_empty``. Complex numbers are a compound type with ``real`` and ``imag``.
A cell is a dataset of object references into the group ``#refs#``. A scalar struct is a group
with one member per field; a struct array stores each field as a dataset of references. A sparse
matrix is a group with ``data``, ``ir`` and ``jc`` and the attribute ``MATLAB_sparse``. The file starts
with a 512-byte user block holding a MAT-file header text.

h5py is installed with mylab. In the browser app it is loaded when a program first uses HDF5.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def h5py_module(what: str = "Version 7.3 MAT-files (they are HDF5 files)"):
    try:
        import h5py
    except ImportError:
        import sys
        if sys.platform == "emscripten":
            raise MatlabError(f"{what} need the Python package 'h5py', which the browser version of mylab could "
                              "not load.", "mylab:hdf5:notInstalled") from None
        raise MatlabError(f"{what} need the Python package 'h5py', which is normally installed with mylab but "
                          "is missing from this Python installation. Install it with 'pip install h5py'.",
                          "mylab:hdf5:notInstalled") from None
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
        _class_attribute(self.h5, obj, cls)
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
            if v.data.size == 1 and not fields:  # struct(): a scalar struct without fields is an empty group
                self._attrs(parent.create_group(name), "struct")
                return
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


def _class_attribute(h5, obj, cls: str) -> None:
    """MATLAB_class as MATLAB writes it: a scalar fixed-length ASCII string, null-terminated type."""
    raw = cls.encode("ascii")
    tid = h5.h5t.C_S1.copy()
    tid.set_size(len(raw))
    tid.set_strpad(h5.h5t.STR_NULLTERM)
    if "MATLAB_class" in obj.attrs:
        del obj.attrs["MATLAB_class"]
    aid = h5.h5a.create(obj.id, b"MATLAB_class", tid, h5.h5s.create(h5.h5s.SCALAR))
    aid.write(np.array(raw, dtype=f"S{len(raw)}"), mtype=tid)  # the same type in memory: no conversion


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
                fields = self._fields(obj) or []
                data = np.empty(shape, dtype=object)
                for idx in np.ndindex(shape):  # a struct array without fields that is not empty
                    data[idx] = {f: np.zeros((0, 0)) for f in fields}
                return Struct(data, fields)
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


# -- parts of variables, for matfile ---------------------------------------------------------------
# A numeric, char or logical variable is one dataset whose C-order element sequence is the column-major
# sequence of the MATLAB array (the file stores the dimensions reversed). Reading m.x(i, j) selects the
# matching hyperslab of that dataset, so only the requested block is read from the file.

_PLAIN = set(_DTYPES) | {"char", "logical"}
_ITEM_BYTES = {"double": 8, "single": 4, "int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "int32": 4, "uint32": 4,
               "int64": 8, "uint64": 8, "char": 2, "logical": 1}


def _matlab_class(obj) -> str:
    cls = obj.attrs.get("MATLAB_class", b"")
    return cls.decode("ascii", errors="replace") if isinstance(cls, bytes) else str(cls)


def _is_refs(h5, ds) -> bool:
    return ds.dtype == h5.ref_dtype or (ds.dtype.kind == "O" and h5.check_ref_dtype(ds.dtype) is not None)


def _is_empty(obj) -> bool:
    return "MATLAB_empty" in obj.attrs and bool(int(np.asarray(obj.attrs["MATLAB_empty"]).flat[0]))


def _matlab_shape(file_shape: tuple) -> tuple:
    if len(file_shape) == 0:
        return (1, 1)
    if len(file_shape) == 1:
        return (1, int(file_shape[0]))
    return tuple(int(n) for n in file_shape[::-1])


def _is_complex(ds) -> bool:
    return bool(ds.dtype.names) and {"real", "imag"} <= set(ds.dtype.names)


def _describe(h5, f, obj, depth: int = 0) -> dict | None:
    """size, class, bytes, complex and sparse of a stored value from its metadata; None when unsure."""
    if depth > 20:
        return None
    cls = _matlab_class(obj)
    if isinstance(obj, h5.Group):
        if "MATLAB_sparse" in obj.attrs:
            rows = int(np.asarray(obj.attrs["MATLAB_sparse"]).flat[0])
            cols = int(obj["jc"].size) - 1
            nnz = int(obj["data"].size) if "data" in obj else 0
            cplx = "data" in obj and _is_complex(obj["data"])
            per = 1 if cls == "logical" else (16 if cplx else 8)
            return {"size": (rows, cols), "class": cls or "double", "bytes": max(nnz, 1) * (per + 8) + (cols + 1) * 8,
                    "complex": cplx, "sparse": True}
        if cls != "struct":
            return None
        fields = [x for x in (_Reader(h5, f, None)._fields(obj) or list(obj.keys())) if x in obj]
        total = 64 * len(fields)
        shape: tuple = (1, 1)
        for x in fields:
            child = obj[x]
            if isinstance(child, h5.Dataset) and "MATLAB_class" not in child.attrs and _is_refs(h5, child):
                shape = _matlab_shape(child.shape)
                part = _refs_bytes(h5, f, child, depth)
            else:
                d = _describe(h5, f, child, depth + 1)
                part = None if d is None else 104 + d["bytes"]
            if part is None:
                return None
            total += part
        return {"size": shape, "class": "struct", "bytes": total, "complex": False, "sparse": False}
    if _is_empty(obj):
        dims = tuple(int(x) for x in np.asarray(obj[()]).flatten())
        if cls not in _PLAIN and cls not in ("cell", "struct"):
            return None
        return {"size": dims if len(dims) >= 2 else (0, 0), "class": cls, "bytes": 0, "complex": False,
                "sparse": False}
    if cls == "canonical empty":
        return {"size": (0, 0), "class": "double", "bytes": 0, "complex": False, "sparse": False}
    if cls == "cell" and _is_refs(h5, obj):
        total = _refs_bytes(h5, f, obj, depth)
        if total is None:
            return None
        return {"size": _matlab_shape(obj.shape), "class": "cell", "bytes": total, "complex": False, "sparse": False}
    if cls in _PLAIN and not _is_refs(h5, obj):
        shape = _matlab_shape(obj.shape)
        cplx = _is_complex(obj)
        return {"size": shape, "class": cls, "bytes": int(np.prod(shape)) * _ITEM_BYTES[cls] * (2 if cplx else 1),
                "complex": cplx, "sparse": False}
    return None


def _refs_bytes(h5, f, ds, depth: int) -> int | None:
    total = 0
    for r in np.asarray(ds[()], dtype=object).reshape(-1):
        d = _describe(h5, f, f[r], depth + 1) if r else {"bytes": 0}
        if d is None:
            return None
        total += 104 + d["bytes"]
    return total


def describe(path: str, name: str) -> dict | None:
    """What whos and size report about one variable, read from the metadata only (None: read the value)."""
    h5 = h5py_module()
    with h5.File(path, "r") as f:
        if name.startswith("#") or name not in f:
            return None
        return _describe(h5, f, f[name])


def _plain_dataset(h5, f, name: str):
    """The dataset of a non-empty numeric, char or logical variable, else None."""
    if name.startswith("#") or name not in f:
        return None
    ds = f[name]
    if not isinstance(ds, h5.Dataset) or _is_empty(ds) or _is_refs(h5, ds) or _matlab_class(ds) not in _PLAIN:
        return None
    return ds


def _to_matlab(raw: np.ndarray, cls: str) -> np.ndarray:
    """Stored values (any shape) converted to the MATLAB class, keeping the shape."""
    a = np.asarray(raw)
    if a.dtype.names and {"real", "imag"} <= set(a.dtype.names):
        a = a["real"] + 1j * a["imag"]
    if cls == "char":
        return np.ascontiguousarray(a.astype("<u4")).view("<U1").reshape(a.shape)
    if cls == "logical":
        return a != 0
    if a.dtype.kind != "c":
        return a.astype(_DTYPES[cls])
    return a.astype(np.complex64 if cls == "single" else np.complex128)


def _to_stored(a: np.ndarray, cls: str, dtype) -> np.ndarray:
    if cls == "char":
        return np.ascontiguousarray(a.astype("<U1")).view("<u4").reshape(a.shape).astype(np.uint16)
    if cls == "logical":
        return a.astype(np.uint8)
    if dtype.names:
        out = np.empty(a.shape, dtype=dtype)
        out["real"] = a.real
        out["imag"] = a.imag
        return out
    return a.astype(dtype)


def _selection(args: list, mshape: tuple):
    """Per MATLAB dimension: (slice of the stored data, positions within that slice); None: read all.

    The slice covers the requested positions, with their step when they are evenly spaced, so a range
    a:s:b reads only the elements asked for. Raises MATLAB's index errors for bad or too large indices.
    """
    from .. import indexing as ix
    k = len(args)
    eshape = ix._effective_shape(mshape, k)
    if k < len(mshape) and eshape[k - 1] != mshape[k - 1]:
        return None  # trailing dimensions folded into the last index
    out = []
    for d in range(k):
        try:
            pos, _ = ix._to_positions(args[d], eshape[d], for_set=False)
        except MatlabError:
            ix.get(np.broadcast_to(np.zeros((), dtype=bool), mshape), args)  # MATLAB's message for the error
            raise
        if pos.size == 0:
            return None
        lo, hi = int(pos.min()), int(pos.max())
        step = 1
        if pos.size > 1:
            diffs = np.diff(pos)
            if diffs[0] > 0 and np.all(diffs == diffs[0]):
                step = int(diffs[0])
        out.append((slice(lo, hi + 1, step), (pos - lo) // step))
    for _ in range(k, len(mshape)):
        out.append((slice(0, 1, 1), np.zeros(1, dtype=np.int64)))
    return out


def _file_index(sel: list, file_ndim: int) -> tuple:
    """h5py index of the stored dataset from per-MATLAB-dimension slices (the dimensions reversed)."""
    if file_ndim == 1:
        return (sel[1][0],)
    return tuple(s for s, _ in sel[:file_ndim])[::-1]


def _block_shape(sel: list) -> tuple:
    return tuple(len(range(s.start, s.stop, s.step)) for s, _ in sel)


def read_part(path: str, name: str, args: list) -> Any:
    """name(args...) of a numeric, char or logical variable, reading only the needed part of the file.

    Returns None when the variable is of another kind or the indexing form is not handled here; the
    caller then reads the whole variable.
    """
    from .. import indexing as ix
    h5 = h5py_module()
    with h5.File(path, "r") as f:
        ds = _plain_dataset(h5, f, name)
        if ds is None or not args:
            return None
        cls = _matlab_class(ds)
        mshape = _matlab_shape(ds.shape)
        if len(args) == 1:
            if args[0] is ix.COLON:
                return None
            n = int(np.prod(mshape))
            pos, ishape = ix._to_positions(args[0], n, for_set=False)
            if pos.size == 0 or ishape is None:
                return None
            lo = int(pos.min())
            vals = _linear_read(ds, lo, int(pos.max()))[pos - lo]
            shape = ix._linear_result_shape(mshape, ishape, pos.size)
            return ix.canonical(np.ascontiguousarray(_to_matlab(vals, cls).reshape(shape, order="F")))
        sel = _selection(args, mshape)
        if sel is None:
            return _folded_read(ds, cls, args, mshape)
        raw = ds[_file_index(sel, ds.ndim)] if ds.ndim else ds[()]
        block = np.asarray(raw).transpose().reshape(_block_shape(sel))
        picked = block[np.ix_(*[p for _, p in sel])]
        return ix.canonical(np.ascontiguousarray(_to_matlab(picked, cls)))


def _folded_read(ds, cls: str, args: list, mshape: tuple) -> Any:
    """A(i, j) with fewer indices than dimensions: the last index runs over the trailing dimensions
    together. The elements are found by their column-major positions."""
    from .. import indexing as ix
    eshape = ix._effective_shape(mshape, len(args))
    pos = [ix._to_positions(a, n, for_set=False)[0] for a, n in zip(args, eshape)]
    if any(p.size == 0 for p in pos):
        return None
    linear = np.ravel_multi_index(np.ix_(*pos), eshape, order="F")
    lo, hi = int(linear.min()), int(linear.max())
    vals = _linear_read(ds, lo, hi)[linear - lo]
    return ix.canonical(np.ascontiguousarray(_to_matlab(vals, cls)))


def _linear_region(ds, lo: int, hi: int) -> tuple[tuple, int]:
    """h5py index covering elements lo..hi (0-based, MATLAB's column-major order), and the offset of
    its first element. Singleton dimensions do not change the order, so a vector reads only lo..hi."""
    shape = [int(n) for n in ds.shape]
    keep = [d for d, n in enumerate(shape) if n != 1]
    index: list = [slice(0, 1)] * len(shape)
    if not keep:
        return tuple(index), 0
    slab = int(np.prod([shape[d] for d in keep[1:]], dtype=np.int64))
    first, last = lo // slab, hi // slab
    for d in keep[1:]:
        index[d] = slice(None)
    index[keep[0]] = slice(first, last + 1)
    return tuple(index), first * slab


def _linear_read(ds, lo: int, hi: int) -> np.ndarray:
    if not ds.shape:
        return np.asarray(ds[()]).reshape(-1)
    index, offset = _linear_region(ds, lo, hi)
    return np.asarray(ds[index]).reshape(-1)[lo - offset:hi - offset + 1]


def _linear_write(ds, lo: int, hi: int, values: np.ndarray) -> None:
    if not ds.shape:
        ds[()] = values.reshape(())
        return
    index, offset = _linear_region(ds, lo, hi)
    current = np.asarray(ds[index])
    flat = current.reshape(-1).copy()
    flat[lo - offset:hi - offset + 1] = values
    ds[index] = flat.reshape(current.shape)


def _selection_for_set(args: list, mshape: tuple):
    from .. import indexing as ix
    eshape = ix._effective_shape(mshape, len(args))
    for d, a in enumerate(args):
        if a is not ix.COLON:
            pos, _ = ix._to_positions(a, eshape[d], for_set=True)
            if pos.size and pos.max() >= eshape[d]:
                return None  # the variable grows: it is rewritten as a whole
    return _selection(args, mshape)


def write_part(path: str, name: str, args: list, value: np.ndarray) -> bool:
    """name(args...) = value, in place, when the selection lies inside the variable and its class stays.

    Returns False, having changed nothing, when the whole variable must be rewritten instead.
    """
    from .. import indexing as ix
    from ..interpreter import coerce_for_assignment
    h5 = h5py_module()
    with h5.File(path, "r+") as f:
        ds = _plain_dataset(h5, f, name)
        if ds is None or not args:
            return False
        cls = _matlab_class(ds)
        mshape = _matlab_shape(ds.shape)
        kind = "<U1" if cls == "char" else bool if cls == "logical" else \
            (np.complex64 if cls == "single" else np.complex128) if _is_complex(ds) else _DTYPES[cls]
        probe = np.zeros((1, 1), dtype=kind)
        cur, val = coerce_for_assignment(probe, value)
        if cur.dtype != probe.dtype:
            return False  # the variable would change class
        try:
            if len(args) == 1:
                if args[0] is ix.COLON:
                    return False
                n = int(np.prod(mshape))
                pos, _ = ix._to_positions(args[0], n, for_set=True)
                if pos.size == 0 or pos.max() >= n or val.size not in (1, pos.size):
                    return False
                lo, hi = int(pos.min()), int(pos.max())
                flat = _to_matlab(_linear_read(ds, lo, hi), cls).copy()
                flat[pos - lo] = val.reshape(-1, order="F")
                _linear_write(ds, lo, hi, _to_stored(flat, cls, ds.dtype))
                return True
            sel = _selection_for_set(args, mshape)
            if sel is None:
                return False
            index = _file_index(sel, ds.ndim) if ds.ndim else ()
            raw = np.asarray(ds[index] if ds.ndim else ds[()])
            bshape = _block_shape(sel)
            block = np.array(_to_matlab(raw, cls).transpose().reshape(bshape))
            local = [np.asarray(p, dtype=float) + 1 for _, p in sel]
            new = ix.set_(block, local, val, lambda: 0)
            if new.size != block.size:
                return False
            stored = _to_stored(new.reshape(bshape[:len(mshape)]).transpose(), cls, ds.dtype).reshape(raw.shape)
            if ds.ndim:
                ds[index] = stored
            else:
                ds[()] = stored
            return True
        except MatlabError:
            return False  # the general path reports the error
