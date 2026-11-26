"""The low-level HDF5 interface: H5F, H5D, H5S, H5G, H5A, H5T and H5ML functions, through h5py.

These follow the HDF5 C library call by call, with MATLAB's conventions: identifiers are opaque
``H5ML.id`` handles, constants are given by name ('H5F_ACC_RDONLY', 'H5P_DEFAULT', 'H5S_ALL', ...)
and offsets are zero-based. Dimensions passed to and returned by the H5S functions are in the C
library's order (slowest first); a MATLAB array read or written with H5D.read, H5D.write, H5A.read
or H5A.write has those dimensions reversed, so a dataspace of dimensions [3 5] holds a 5-by-3 array.
Property lists other than 'H5P_DEFAULT' are not supported.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, StringArray, char
from ._registry import builtin
from ._util import is_text, text


def _h5():
    from ._mat73 import h5py_module
    return h5py_module("The low-level HDF5 functions")


class H5Id(MObject):
    """An HDF5 identifier: a file, group, dataset, attribute, dataspace or datatype."""

    is_handle = True

    def __init__(self, kind: str, obj: Any, path: str = ""):
        self.kind = kind
        self.obj = obj
        self.path = path
        self.selection: tuple | None = None  # (count * block) in C order after a SET hyperslab selection

    def class_name(self) -> str:
        return "H5ML.id"

    def isa_class(self, name: str) -> bool:
        return name in ("H5ML.id", "handle")

    def get(self, kind: str | tuple, fn: str):
        kinds = (kind,) if isinstance(kind, str) else kind
        if self.obj is None:
            raise MatlabError(f"{fn}: the identifier has been closed.", "MATLAB:imagesci:hdf5lib:libraryError")
        if self.kind not in kinds:
            raise MatlabError(f"{fn}: expected a {' or '.join(kinds)} identifier, not a {self.kind} identifier.")
        return self.obj

    def close(self) -> None:
        obj, self.obj = self.obj, None
        if obj is None:
            return
        closer = getattr(obj, "close", None) if self.kind == "file" else getattr(obj, "_close", None)
        if closer is not None:
            try:
                closer()
            except (ValueError, RuntimeError, OSError):
                pass

    def display_lines(self) -> list[str]:
        state = "closed" if self.obj is None else "open"
        where = f", {self.path}" if self.path else ""
        return [f"  H5ML.id ({self.kind} identifier, {state}{where})"]


def _id(v: Any, kind: str | tuple, fn: str):
    if not isinstance(v, H5Id):
        kinds = (kind,) if isinstance(kind, str) else kind
        raise MatlabError(f"{fn}: expected a {' or '.join(kinds)} identifier.")
    return v.get(kind, fn)


def _nargs(args: list, allowed: tuple, fn: str, usage: str) -> None:
    if len(args) not in allowed:
        raise MatlabError(f"{fn}: wrong number of arguments; use {usage}.", "MATLAB:narginchk:notEnoughInputs"
                          if len(args) < min(allowed) else "MATLAB:narginchk:tooManyInputs")


def _bname(v: Any, what: str) -> bytes:
    return text(v, what).encode("utf-8")


def _default_plist(v: Any, fn: str) -> None:
    if is_text(v) and text(v) == "H5P_DEFAULT":
        return
    if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "iuf" and float(v.flat[0]) == 0:
        return  # H5ML.get_constant_value('H5P_DEFAULT')
    raise MatlabError(f"{fn}: only the default property list 'H5P_DEFAULT' is supported.")


# -- constants and predefined types ------------------------------------------------------------------

def _constants(h5) -> dict:
    return {"H5F_ACC_RDONLY": h5.h5f.ACC_RDONLY, "H5F_ACC_RDWR": h5.h5f.ACC_RDWR, "H5F_ACC_TRUNC": h5.h5f.ACC_TRUNC,
            "H5F_ACC_EXCL": h5.h5f.ACC_EXCL, "H5P_DEFAULT": 0, "H5S_ALL": 0, "H5ML_DEFAULT": -2,
            "H5S_UNLIMITED": -1, "H5S_SELECT_SET": h5.h5s.SELECT_SET, "H5S_SELECT_OR": h5.h5s.SELECT_OR,
            "H5S_SELECT_AND": h5.h5s.SELECT_AND, "H5S_SELECT_XOR": h5.h5s.SELECT_XOR,
            "H5S_SELECT_NOTB": h5.h5s.SELECT_NOTB, "H5S_SELECT_NOTA": h5.h5s.SELECT_NOTA,
            "H5S_SCALAR": h5.h5s.SCALAR, "H5S_SIMPLE": h5.h5s.SIMPLE, "H5S_NULL": h5.h5s.NULL,
            "H5T_VARIABLE": -3}


def _flag(h5, v: Any, allowed: tuple, fn: str) -> int:
    consts = _constants(h5)
    if is_text(v):
        name = text(v)
        if name not in allowed:
            raise MatlabError(f"{fn}: '{name}' is not allowed here; use {', '.join(repr(a) for a in allowed)}.")
        return consts[name]
    if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "iuf":
        x = int(v.flat[0])
        if x in [consts[a] for a in allowed]:
            return x
    raise MatlabError(f"{fn}: expected one of {', '.join(repr(a) for a in allowed)}.")


# C type names of the native types (long is 64 bits, as on 64-bit Linux and macOS)
_NATIVE_ALIASES = {"NATIVE_CHAR": "NATIVE_INT8", "NATIVE_SCHAR": "NATIVE_INT8", "NATIVE_UCHAR": "NATIVE_UINT8",
                   "NATIVE_SHORT": "NATIVE_INT16", "NATIVE_USHORT": "NATIVE_UINT16", "NATIVE_INT": "NATIVE_INT32",
                   "NATIVE_UINT": "NATIVE_UINT32", "NATIVE_LONG": "NATIVE_INT64", "NATIVE_ULONG": "NATIVE_UINT64",
                   "NATIVE_LLONG": "NATIVE_INT64", "NATIVE_ULLONG": "NATIVE_UINT64", "NATIVE_HSIZE": "NATIVE_UINT64",
                   "NATIVE_HSSIZE": "NATIVE_INT64", "NATIVE_HERR": "NATIVE_INT32", "NATIVE_HBOOL": "NATIVE_UINT8"}


def _predefined(h5, name: str, fn: str):
    if name.startswith("H5T_") and name[4:] in _NATIVE_ALIASES:
        name = "H5T_" + _NATIVE_ALIASES[name[4:]]
    if not name.startswith("H5T_") or not hasattr(h5.h5t, name[4:]):
        raise MatlabError(f"{fn}: '{name}' is not a predefined HDF5 datatype name such as 'H5T_NATIVE_DOUBLE'.")
    tid = getattr(h5.h5t, name[4:])
    if not isinstance(tid, h5.h5t.TypeID):
        raise MatlabError(f"{fn}: '{name}' is not a predefined HDF5 datatype name.")
    return tid


def _type(h5, v: Any, fn: str, allow_default: bool = False):
    """A datatype argument: a predefined name, an identifier, or (if allowed) 'H5ML_DEFAULT' -> None."""
    if is_text(v):
        name = text(v)
        if name == "H5ML_DEFAULT" and allow_default:
            return None
        return _predefined(h5, name, fn)
    if allow_default and isinstance(v, np.ndarray) and v.size == 1 and float(v.flat[0]) == -2:
        return None
    return _id(v, "datatype", fn)


def _space(h5, v: Any, fn: str):
    """A dataspace argument: 'H5S_ALL' -> (h5s.ALL, None) or an identifier -> (space, its H5Id)."""
    if (is_text(v) and text(v) == "H5S_ALL") or (isinstance(v, np.ndarray) and v.size == 1 and float(v.flat[0]) == 0):
        return h5.h5s.ALL, None
    return _id(v, "dataspace", fn), v


def _dims(v: Any, what: str) -> tuple:
    a = np.asarray(v, dtype=float).reshape(-1)
    if any(x < 0 or x != int(x) for x in a if np.isfinite(x)):
        raise MatlabError(f"{what} must be non-negative integers.")
    return tuple(int(x) for x in a)


# -- converting data between MATLAB and the C layout ---------------------------------------------------

def _to_matlab(h5, raw: np.ndarray, dtype) -> Any:
    """C-order data (dimensions slowest first) as a MATLAB value with the dimensions reversed."""
    a = np.asarray(raw)
    info = h5.check_string_dtype(dtype) if dtype is not None else None
    if a.dtype.kind == "O" or (info is not None and info.length is None):  # variable-length strings: a cell
        flat = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in a.reshape(-1)]
        if a.ndim == 0:
            return char(flat[0])
        out = np.empty(len(flat), dtype=object)
        for k, s in enumerate(flat):
            out[k] = char(s)
        shape = a.shape[::-1] if a.ndim > 1 else (a.size, 1)
        return Cell(out.reshape(a.shape).transpose().reshape(shape))
    if a.dtype.kind == "S":  # fixed-length strings: each string is a column of characters
        width = max(a.dtype.itemsize, 1)
        codes = np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=np.uint8).reshape(a.shape + (width,))
        chars = np.ascontiguousarray(codes.astype("<u4")).view("<U1").reshape(codes.shape)
        if a.ndim == 0:
            return np.ascontiguousarray(chars.reshape(1, -1))
        return np.ascontiguousarray(chars.transpose())
    if a.dtype.names:
        from ..values import Struct
        return Struct.scalar({n: _to_matlab(h5, a[n], a.dtype[n]) for n in a.dtype.names})
    if a.dtype.kind not in "biuf":
        raise MatlabError(f"HDF5 data of type {a.dtype} cannot be converted to a MATLAB value.")
    a = a.astype(a.dtype.newbyteorder("="))
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return np.ascontiguousarray(a.reshape(-1, 1))
    return np.ascontiguousarray(a.transpose())


def _from_matlab(h5, v: Any, shape: tuple, dtype, fn: str) -> np.ndarray:
    """A MATLAB value as a C-order array of the given shape (dimensions reversed) and type."""
    info = h5.check_string_dtype(dtype)
    if info is not None or dtype.kind == "S":
        if isinstance(v, (Cell, StringArray, MString)):
            items = [text(x) for x in v.data.flatten(order="F")] if isinstance(v, Cell) else \
                [str(v)] if isinstance(v, MString) else ["" if x is None else x for x in v.data.flatten(order="F")]
        elif isinstance(v, np.ndarray) and v.dtype.kind == "U":
            cols = v.reshape(v.shape[0], -1, order="F") if v.ndim > 1 else v.reshape(-1, 1)
            items = ["".join(cols[:, k].tolist()).rstrip("\0") for k in range(cols.shape[1])] \
                if (info is None or info.length is not None) and int(np.prod(shape) or 1) == cols.shape[1] \
                else ["".join(v.reshape(-1, order="F").tolist())]
        else:
            raise MatlabError(f"{fn}: the datatype holds text, so the data must be text.")
        if info is not None and info.length is None:
            arr = np.array(items, dtype=object)
        else:
            arr = np.array([s.encode("utf-8") for s in items], dtype=dtype)
        return arr.reshape(shape) if arr.size == int(np.prod(shape)) else arr
    if isinstance(v, MString):
        v = char(str(v))
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufU":
        raise MatlabError(f"{fn}: the data must be a numeric, logical or char array.")
    if v.dtype.kind == "U":
        from ..values import char_codes
        v = char_codes(v)
    a = np.ascontiguousarray(v.transpose())  # reversed dimensions, C order = MATLAB's column-major order
    if a.size != int(np.prod(shape, dtype=np.int64)):
        raise MatlabError(f"{fn}: the data has {a.size} elements but the selection holds "
                          f"{int(np.prod(shape, dtype=np.int64))}.")
    if dtype.names or dtype.kind not in "biuf":
        raise MatlabError(f"{fn}: writing to a datatype of class {dtype} is not supported.")
    return a.reshape(shape).astype(dtype)


def _selection_shape(space_id: H5Id | None, space) -> tuple:
    """Shape (C order) of what a file dataspace selects, for a memory space made to match it."""
    if space_id is not None and space_id.selection is not None:
        return space_id.selection
    if space.get_select_type() == _h5().h5s.SEL_ALL:
        return tuple(space.shape)
    return (int(space.get_select_npoints()),)


def _io_spaces(h5, obj_shape: tuple, mspace_arg: Any, fspace_arg: Any, fn: str):
    fspace, fsid = _space(h5, fspace_arg, fn)
    mspace, msid = _space(h5, mspace_arg, fn)
    if mspace is not h5.h5s.ALL:
        return mspace, fspace, tuple(mspace.shape)
    if fspace is h5.h5s.ALL:
        return mspace, fspace, tuple(obj_shape)
    shape = _selection_shape(fsid, fspace)
    return h5.h5s.create_simple(shape), fspace, shape


# -- H5F --------------------------------------------------------------------------------------------

@builtin("H5F.open")
def _h5f_open(interp, args, nargout):
    _nargs(args, (1, 2, 3), "H5F.open", "fid = H5F.open(filename, flags, fapl)")
    h5 = _h5()
    flags = _flag(h5, args[1], ("H5F_ACC_RDONLY", "H5F_ACC_RDWR"), "H5F.open") if len(args) > 1 \
        else h5.h5f.ACC_RDONLY
    if len(args) > 2:
        _default_plist(args[2], "H5F.open")
    from .fileformats import find_file
    name = os.path.expanduser(text(args[0], "the file name"))
    path = find_file(interp, name) or name
    if not os.path.isfile(path):
        raise MatlabError(f"H5F.open: the file '{name}' does not exist.", "MATLAB:imagesci:hdf5lib:fileOpenErr")
    try:
        return H5Id("file", h5.h5f.open(os.path.abspath(path).encode("utf-8"), flags), os.path.abspath(path))
    except OSError as e:
        raise MatlabError(f"H5F.open: unable to open '{name}' ({e}).", "MATLAB:imagesci:hdf5lib:fileOpenErr") \
            from None


@builtin("H5F.create")
def _h5f_create(interp, args, nargout):
    _nargs(args, (1, 2, 3, 4), "H5F.create", "fid = H5F.create(filename, flags, fcpl, fapl)")
    h5 = _h5()
    flags = _flag(h5, args[1], ("H5F_ACC_TRUNC", "H5F_ACC_EXCL"), "H5F.create") if len(args) > 1 \
        else h5.h5f.ACC_EXCL
    for p in args[2:]:
        _default_plist(p, "H5F.create")
    path = os.path.abspath(os.path.expanduser(text(args[0], "the file name")))
    try:
        return H5Id("file", h5.h5f.create(path.encode("utf-8"), flags), path)
    except OSError as e:
        raise MatlabError(f"H5F.create: unable to create '{path}' ({e}).",
                          "MATLAB:imagesci:hdf5lib:fileCreateErr") from None


def _closer(name: str, kind: str | tuple):
    @builtin(name)
    def close(interp, args, nargout):
        _nargs(args, (1,), name, f"{name}(id)")
        if not isinstance(args[0], H5Id):
            raise MatlabError(f"{name}: expected an identifier.")
        args[0].get(kind, name)
        args[0].close()
        return None
    return close


_closer("H5F.close", "file")
_closer("H5D.close", "dataset")
_closer("H5S.close", "dataspace")
_closer("H5G.close", "group")
_closer("H5A.close", "attribute")
_closer("H5T.close", "datatype")


# -- H5G --------------------------------------------------------------------------------------------

@builtin("H5G.open")
def _h5g_open(interp, args, nargout):
    _nargs(args, (2, 3), "H5G.open", "gid = H5G.open(loc_id, name, gapl)")
    h5 = _h5()
    loc = _id(args[0], ("file", "group"), "H5G.open")
    if len(args) > 2:
        _default_plist(args[2], "H5G.open")
    try:
        return H5Id("group", h5.h5g.open(loc, _bname(args[1], "the group name")))
    except (KeyError, OSError, ValueError) as e:
        raise MatlabError(f"H5G.open: unable to open the group '{text(args[1])}' ({e}).",
                          "MATLAB:imagesci:hdf5lib:libraryError") from None


@builtin("H5G.create")
def _h5g_create(interp, args, nargout):
    _nargs(args, (2, 3, 5), "H5G.create", "gid = H5G.create(loc_id, name, lcpl, gcpl, gapl)")
    h5 = _h5()
    loc = _id(args[0], ("file", "group"), "H5G.create")
    if len(args) == 5:
        for p in args[2:]:
            _default_plist(p, "H5G.create")
    try:
        return H5Id("group", h5.h5g.create(loc, _bname(args[1], "the group name")))
    except (ValueError, OSError) as e:
        raise MatlabError(f"H5G.create: unable to create the group '{text(args[1])}' ({e}).",
                          "MATLAB:imagesci:hdf5lib:libraryError") from None


# -- H5S --------------------------------------------------------------------------------------------

@builtin("H5S.create_simple")
def _h5s_create_simple(interp, args, nargout):
    _nargs(args, (3,), "H5S.create_simple", "space_id = H5S.create_simple(rank, dims, maxdims)")
    h5 = _h5()
    rank = int(np.asarray(args[0], dtype=float).reshape(-1)[0])
    dims = _dims(args[1], "H5S.create_simple: the dimensions")
    if len(dims) != rank:
        raise MatlabError(f"H5S.create_simple: rank is {rank} but {len(dims)} dimensions were given.")
    mx = np.asarray(args[2], dtype=float).reshape(-1)
    if mx.size == 0:
        maxdims = None
    else:
        if mx.size != rank:
            raise MatlabError("H5S.create_simple: maxdims needs one value per dimension.")
        maxdims = tuple(h5.h5s.UNLIMITED if (x < 0 or np.isinf(x) or x >= 2 ** 63) else int(x) for x in mx)
    try:
        return H5Id("dataspace", h5.h5s.create_simple(dims, maxdims))
    except ValueError as e:
        raise MatlabError(f"H5S.create_simple: {e}") from None


@builtin("H5S.create")
def _h5s_create(interp, args, nargout):
    _nargs(args, (1,), "H5S.create", "space_id = H5S.create(type)")
    h5 = _h5()
    kind = _flag(h5, args[0], ("H5S_SCALAR", "H5S_SIMPLE", "H5S_NULL"), "H5S.create")
    return H5Id("dataspace", h5.h5s.create(kind))


@builtin("H5S.get_simple_extent_dims")
def _h5s_get_simple_extent_dims(interp, args, nargout):
    _nargs(args, (1,), "H5S.get_simple_extent_dims", "[ndims, dims, maxdims] = H5S.get_simple_extent_dims(id)")
    space = _id(args[0], "dataspace", "H5S.get_simple_extent_dims")
    dims = tuple(space.shape)
    mx = space.get_simple_extent_dims(maxdims=True)
    row = (lambda xs: np.array([[float(x) for x in xs]]) if xs else np.zeros((1, 0)))  # noqa: E731
    return (np.array([[float(len(dims))]]), row(dims), row([-1 if m == _h5().h5s.UNLIMITED else m for m in mx]))


@builtin("H5S.select_hyperslab")
def _h5s_select_hyperslab(interp, args, nargout):
    _nargs(args, (6,), "H5S.select_hyperslab", "H5S.select_hyperslab(space_id, op, start, stride, count, block)")
    h5 = _h5()
    space = _id(args[0], "dataspace", "H5S.select_hyperslab")
    op = _flag(h5, args[1], ("H5S_SELECT_SET", "H5S_SELECT_OR", "H5S_SELECT_AND", "H5S_SELECT_XOR",
                             "H5S_SELECT_NOTB", "H5S_SELECT_NOTA"), "H5S.select_hyperslab")
    rank = space.get_simple_extent_ndims()
    start = _dims(args[2], "H5S.select_hyperslab: start")
    count = _dims(args[4], "H5S.select_hyperslab: count")
    stride = _dims(args[3], "H5S.select_hyperslab: stride") or (1,) * rank
    block = _dims(args[5], "H5S.select_hyperslab: block") or (1,) * rank
    if not (len(start) == len(count) == len(stride) == len(block) == rank):
        raise MatlabError(f"H5S.select_hyperslab: start, stride, count and block need {rank} values each.")
    try:
        space.select_hyperslab(start, count, stride, block, op=op)
    except ValueError as e:
        raise MatlabError(f"H5S.select_hyperslab: {e}") from None
    args[0].selection = tuple(c * b for c, b in zip(count, block)) if op == h5.h5s.SELECT_SET else None
    return None


# -- H5T --------------------------------------------------------------------------------------------

@builtin("H5T.copy")
def _h5t_copy(interp, args, nargout):
    _nargs(args, (1,), "H5T.copy", "type_id = H5T.copy(type)")
    h5 = _h5()
    if isinstance(args[0], H5Id) and args[0].kind == "dataset":
        return H5Id("datatype", args[0].get("dataset", "H5T.copy").get_type())
    return H5Id("datatype", _type(h5, args[0], "H5T.copy").copy())


@builtin("H5T.set_size")
def _h5t_set_size(interp, args, nargout):
    _nargs(args, (2,), "H5T.set_size", "H5T.set_size(type_id, size)")
    h5 = _h5()
    tid = _id(args[0], "datatype", "H5T.set_size")
    if is_text(args[1]) and text(args[1]) == "H5T_VARIABLE":
        size = h5.h5t.VARIABLE
    else:
        size = int(np.asarray(args[1], dtype=float).reshape(-1)[0])
        if size == -3:
            size = h5.h5t.VARIABLE
    try:
        tid.set_size(size)
    except (ValueError, TypeError) as e:
        raise MatlabError(f"H5T.set_size: {e}") from None
    return None


# -- H5D --------------------------------------------------------------------------------------------

@builtin("H5D.open")
def _h5d_open(interp, args, nargout):
    _nargs(args, (2, 3), "H5D.open", "dset_id = H5D.open(loc_id, name, dapl)")
    h5 = _h5()
    loc = _id(args[0], ("file", "group"), "H5D.open")
    if len(args) > 2:
        _default_plist(args[2], "H5D.open")
    try:
        return H5Id("dataset", h5.h5d.open(loc, _bname(args[1], "the dataset name")))
    except (KeyError, OSError, ValueError) as e:
        raise MatlabError(f"H5D.open: unable to open the dataset '{text(args[1])}' ({e}).",
                          "MATLAB:imagesci:hdf5lib:libraryError") from None


@builtin("H5D.create")
def _h5d_create(interp, args, nargout):
    _nargs(args, (4, 5, 7), "H5D.create", "dset_id = H5D.create(loc_id, name, type_id, space_id, lcpl, dcpl, dapl)")
    h5 = _h5()
    loc = _id(args[0], ("file", "group"), "H5D.create")
    tid = _type(h5, args[2], "H5D.create")
    space = _id(args[3], "dataspace", "H5D.create")
    for p in args[4:]:
        _default_plist(p, "H5D.create")
    try:
        return H5Id("dataset", h5.h5d.create(loc, _bname(args[1], "the dataset name"), tid, space))
    except (ValueError, OSError, TypeError) as e:
        raise MatlabError(f"H5D.create: unable to create the dataset '{text(args[1])}' ({e}). A dataspace with "
                          "unlimited dimensions needs a chunked layout, which needs a dataset creation property "
                          "list; those are not supported.", "MATLAB:imagesci:hdf5lib:libraryError") from None


@builtin("H5D.get_space")
def _h5d_get_space(interp, args, nargout):
    _nargs(args, (1,), "H5D.get_space", "space_id = H5D.get_space(dset_id)")
    return H5Id("dataspace", _id(args[0], "dataset", "H5D.get_space").get_space())


@builtin("H5D.get_type")
def _h5d_get_type(interp, args, nargout):
    _nargs(args, (1,), "H5D.get_type", "type_id = H5D.get_type(dset_id)")
    return H5Id("datatype", _id(args[0], "dataset", "H5D.get_type").get_type())


@builtin("H5D.read")
def _h5d_read(interp, args, nargout):
    _nargs(args, (1, 2, 5), "H5D.read", "data = H5D.read(dset_id, mem_type, mem_space, file_space, dxpl)")
    h5 = _h5()
    dsid = _id(args[0], "dataset", "H5D.read")
    mtype = _type(h5, args[1], "H5D.read", allow_default=True) if len(args) > 1 else None
    if len(args) == 5:
        _default_plist(args[4], "H5D.read")
        mspace, fspace, shape = _io_spaces(h5, dsid.shape, args[2], args[3], "H5D.read")
    else:
        mspace, fspace, shape = h5.h5s.ALL, h5.h5s.ALL, tuple(dsid.shape)
    dtype = dsid.dtype if mtype is None else mtype.dtype
    arr = np.empty(shape, dtype=dtype)
    try:
        dsid.read(mspace, fspace, arr, mtype)
    except (OSError, TypeError, ValueError) as e:
        raise MatlabError(f"H5D.read: {e}", "MATLAB:imagesci:hdf5lib:libraryError") from None
    return _to_matlab(h5, arr, dtype)


@builtin("H5D.write")
def _h5d_write(interp, args, nargout):
    _nargs(args, (6,), "H5D.write", "H5D.write(dset_id, mem_type, mem_space, file_space, dxpl, data)")
    h5 = _h5()
    dsid = _id(args[0], "dataset", "H5D.write")
    mtype = _type(h5, args[1], "H5D.write", allow_default=True)
    _default_plist(args[4], "H5D.write")
    mspace, fspace, shape = _io_spaces(h5, dsid.shape, args[2], args[3], "H5D.write")
    dtype = dsid.dtype if mtype is None else mtype.dtype
    arr = _from_matlab(h5, args[5], shape, dtype, "H5D.write")
    if mtype is None and arr.dtype.kind == "S":
        mtype = dsid.get_type()
    try:
        dsid.write(mspace, fspace, arr, mtype)
    except (OSError, TypeError, ValueError) as e:
        raise MatlabError(f"H5D.write: {e}", "MATLAB:imagesci:hdf5lib:libraryError") from None
    return None


# -- H5A --------------------------------------------------------------------------------------------

_HOLDERS = ("file", "group", "dataset")


@builtin("H5A.open")
def _h5a_open(interp, args, nargout):
    _nargs(args, (2, 3), "H5A.open", "attr_id = H5A.open(obj_id, attr_name, aapl)")
    h5 = _h5()
    obj = _id(args[0], _HOLDERS, "H5A.open")
    if len(args) > 2:
        _default_plist(args[2], "H5A.open")
    try:
        return H5Id("attribute", h5.h5a.open(obj, _bname(args[1], "the attribute name")))
    except (KeyError, OSError, ValueError) as e:
        raise MatlabError(f"H5A.open: unable to open the attribute '{text(args[1])}' ({e}).",
                          "MATLAB:imagesci:hdf5lib:libraryError") from None


@builtin("H5A.create")
def _h5a_create(interp, args, nargout):
    _nargs(args, (4, 5, 6), "H5A.create", "attr_id = H5A.create(loc_id, name, type_id, space_id, acpl, aapl)")
    h5 = _h5()
    obj = _id(args[0], _HOLDERS, "H5A.create")
    tid = _type(h5, args[2], "H5A.create")
    space = _id(args[3], "dataspace", "H5A.create")
    for p in args[4:]:
        _default_plist(p, "H5A.create")
    try:
        return H5Id("attribute", h5.h5a.create(obj, _bname(args[1], "the attribute name"), tid, space))
    except (ValueError, OSError, TypeError) as e:
        raise MatlabError(f"H5A.create: unable to create the attribute '{text(args[1])}' ({e}).",
                          "MATLAB:imagesci:hdf5lib:libraryError") from None


@builtin("H5A.read")
def _h5a_read(interp, args, nargout):
    _nargs(args, (1, 2), "H5A.read", "data = H5A.read(attr_id, mem_type)")
    h5 = _h5()
    aid = _id(args[0], "attribute", "H5A.read")
    mtype = _type(h5, args[1], "H5A.read", allow_default=True) if len(args) > 1 else None
    dtype = aid.dtype if mtype is None else mtype.dtype
    arr = np.empty(aid.shape, dtype=dtype)
    try:
        aid.read(arr, mtype)
    except (OSError, TypeError, ValueError) as e:
        raise MatlabError(f"H5A.read: {e}", "MATLAB:imagesci:hdf5lib:libraryError") from None
    return _to_matlab(h5, arr, dtype)


@builtin("H5A.write")
def _h5a_write(interp, args, nargout):
    _nargs(args, (3,), "H5A.write", "H5A.write(attr_id, mem_type, data)")
    h5 = _h5()
    aid = _id(args[0], "attribute", "H5A.write")
    mtype = _type(h5, args[1], "H5A.write", allow_default=True)
    dtype = aid.dtype if mtype is None else mtype.dtype
    arr = _from_matlab(h5, args[2], tuple(aid.shape), dtype, "H5A.write")
    if mtype is None and arr.dtype.kind == "S":
        mtype = aid.get_type()  # the bytes as they are: no conversion that would make room for a terminator
    try:
        aid.write(arr, mtype)
    except (OSError, TypeError, ValueError) as e:
        raise MatlabError(f"H5A.write: {e}", "MATLAB:imagesci:hdf5lib:libraryError") from None
    return None


# -- H5ML -------------------------------------------------------------------------------------------

@builtin("H5ML.get_constant_value")
def _h5ml_get_constant_value(interp, args, nargout):
    _nargs(args, (1,), "H5ML.get_constant_value", "value = H5ML.get_constant_value(name)")
    name = text(args[0], "the constant name")
    consts = _constants(_h5())
    if name not in consts:
        raise MatlabError(f"H5ML.get_constant_value: unknown constant '{name}'. Known names: "
                          f"{', '.join(sorted(consts))}.")
    return np.array([[float(consts[name])]])
