"""NetCDF-4 files: NetCDF data stored in an HDF5 file, read and written through the h5py package.

The storage follows the published NetCDF-4 format conventions, so files written here open in other
NetCDF tools and files from them open here:

* Groups are HDF5 groups; variables are HDF5 datasets, with the dimensions in C order (slowest first).
* A dimension is an HDF5 dimension scale (attributes CLASS and NAME) with an integer ``_Netcdf4Dimid``.
  A dimension that is not also a coordinate variable is a dataset of the dimension's length whose NAME
  begins with "This is a netCDF dimension but not a netCDF variable." An unlimited dimension has an
  unlimited maximum size; its length is the longest extent of the variables that use it.
* Variables are attached to their dimension scales (``DIMENSION_LIST``) and carry ``_Netcdf4Coordinates``,
  the ids of their dimensions.
* The root group has ``_NCProperties``; files in the classic model also have ``_nc3_strict``.
* Text attributes are fixed-length strings (NC_CHAR); string arrays use variable-length strings (NC_STRING).
* The fill value is the HDF5 fill value and, when the user sets one, the ``_FillValue`` attribute.

MATLAB lists dimensions fastest first, the reverse of the file order, as for classic files.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, Struct, char, class_name
from ._util import bool_scalar, is_text, text

DIM_TAG = "This is a netCDF dimension but not a netCDF variable."
_HIDDEN = {"CLASS", "NAME", "REFERENCE_LIST", "DIMENSION_LIST", "_Netcdf4Dimid", "_Netcdf4Coordinates",
           "_NCProperties", "_nc3_strict", "_SuperblockVersion", "_IsNetcdf4"}
_NC_FILL_FLOAT = 9.9692099683868690e+36
# MATLAB class -> (stored numpy type, NetCDF default fill value)
TYPES = {"double": (np.dtype(np.float64), _NC_FILL_FLOAT), "single": (np.dtype(np.float32), _NC_FILL_FLOAT),
         "int8": (np.dtype(np.int8), -127), "uint8": (np.dtype(np.uint8), 255),
         "int16": (np.dtype(np.int16), -32767), "uint16": (np.dtype(np.uint16), 65535),
         "int32": (np.dtype(np.int32), -2147483647), "uint32": (np.dtype(np.uint32), 4294967295),
         "int64": (np.dtype(np.int64), -9223372036854775806),
         "uint64": (np.dtype(np.uint64), 18446744073709551614), "char": (np.dtype("S1"), b"\0")}
CLASSIC_TYPES = ("double", "single", "int32", "int16", "int8", "char")
_ERR = "MATLAB:imagesci:netcdf:libraryFailure"


def _h5():
    from ._mat73 import h5py_module
    return h5py_module("NetCDF-4 files (they are HDF5 files)")


def is_netcdf4(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        return False
    if head == b"\x89HDF\r\n\x1a\n":
        return True
    if head[:3] == b"CDF" or len(head) < 8:
        return False
    try:
        return bool(_h5().is_hdf5(path))  # an HDF5 file with a user block
    except MatlabError:
        return False  # without h5py the classic reader reports what the file is


def open_file(path: str, mode: str = "r"):
    h5 = _h5()
    try:
        return h5, h5.File(path, mode)
    except OSError as e:
        raise MatlabError(f"Unable to open the NetCDF-4 file '{path}': {e}", "MATLAB:imagesci:netcdf:fileOpen") \
            from None


def create_file(path: str, fmt: str):
    h5 = _h5()
    try:
        f = h5.File(path, "w", track_order=True, libver="earliest")
    except OSError as e:
        raise MatlabError(f"Unable to create the NetCDF-4 file '{path}': {e}") from None
    from .. import __version__
    props = f"version=2,mylab={__version__},hdf5={h5.version.hdf5_version}"
    _put_text_attr(h5, f, "_NCProperties", props)
    if fmt == "netcdf4_classic":
        f.attrs.create("_nc3_strict", np.int32(1))
    return h5, f


def file_format(f) -> str:
    return "netcdf4_classic" if "_nc3_strict" in f.attrs else "netcdf4"


# -- names and dimensions ------------------------------------------------------------------------

def _attr_text(obj, name: str) -> str:
    if name not in obj.attrs:
        return ""
    v = obj.attrs[name]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").rstrip("\0")
    return str(v) if isinstance(v, str) else ""


def _is_scale(ds) -> bool:
    return _attr_text(ds, "CLASS") == "DIMENSION_SCALE"


def _is_pure_dim(ds) -> bool:
    return _is_scale(ds) and _attr_text(ds, "NAME").startswith(DIM_TAG)


def _base(name: str) -> str:
    return name.rstrip("/").rsplit("/", 1)[-1]


def split_location(loc: str) -> tuple[str, str]:
    """'/grp/sub/var' -> ('/grp/sub', 'var'); 'var' -> ('/', 'var')."""
    loc = "/" + loc.strip("/")
    group, _, name = loc.rpartition("/")
    return group or "/", name


def find_group(h5, f, path: str, fn: str):
    path = "/" + path.strip("/")
    if path == "/":
        return f
    if path not in f or not isinstance(f[path], h5.Group):
        raise MatlabError(f"{fn}: the NetCDF file has no group '{path}'.", "MATLAB:imagesci:netcdf:unknownLocation")
    return f[path]


def groups_in(h5, g) -> list:
    return [g[k] for k in g.keys() if g.get(k, getclass=True) is h5.Group]


def dims_in(h5, g) -> dict:
    """{name: scale dataset} of the dimensions defined in group g, in order of their ids."""
    found = []
    for k in g.keys():
        if g.get(k, getclass=True) is h5.Dataset:
            ds = g[k]
            if _is_scale(ds):
                dimid = int(np.asarray(ds.attrs["_Netcdf4Dimid"]).flat[0]) if "_Netcdf4Dimid" in ds.attrs \
                    else 1 << 30
                found.append((dimid, len(found), k, ds))
    return {k: ds for _, _, k, ds in sorted(found)}


def vars_in(h5, g) -> dict:
    out = {}
    for k in g.keys():
        if g.get(k, getclass=True) is h5.Dataset:
            ds = g[k]
            if not _is_pure_dim(ds):
                out[k] = ds
    return out


def dim_length(h5, f, scale) -> int:
    if scale.maxshape[0] is not None:
        return int(scale.shape[0])
    n = 0 if _is_pure_dim(scale) else int(scale.shape[0])
    for ref, k in _references(scale):
        try:
            n = max(n, int(f[ref].shape[int(k)]))
        except (KeyError, ValueError, IndexError):
            pass
    return n


def _references(scale) -> list:
    if "REFERENCE_LIST" not in scale.attrs:
        return []
    try:
        items = scale.attrs["REFERENCE_LIST"]
    except (OSError, TypeError, ValueError):
        return []
    return [(row[0], row[1]) for row in np.atleast_1d(items)]


def var_dims(h5, f, ds) -> list:
    """[(name, scale or None, length, unlimited)] of a variable, in file order (slowest first)."""
    out = []
    if ds.ndim == 0:
        return out
    if _is_scale(ds) and ds.ndim == 1:  # a coordinate variable is its own dimension
        return [(_base(ds.name), ds, dim_length(h5, f, ds), ds.maxshape[0] is None)]
    for k in range(ds.ndim):
        scale = None
        try:
            if len(ds.dims[k]):
                scale = ds.dims[k][0]
        except (OSError, RuntimeError, KeyError):
            scale = None
        if scale is not None:
            out.append((_base(scale.name), scale, dim_length(h5, f, scale), scale.maxshape[0] is None))
        else:
            out.append((f"phony_dim_{k}", None, int(ds.shape[k]), ds.maxshape[k] is None))
    return out


def _all_dimids(h5, f) -> list[int]:
    ids: list[int] = []

    def visit(name, obj):
        if isinstance(obj, h5.Dataset) and "_Netcdf4Dimid" in obj.attrs and _is_scale(obj):
            ids.append(int(np.asarray(obj.attrs["_Netcdf4Dimid"]).flat[0]))
    f.visititems(visit)
    return ids


def _next_dimid(h5, f) -> int:
    ids = _all_dimids(h5, f)
    return max(ids) + 1 if ids else 0


def _visible_dim(h5, g, name: str):
    """The dimension 'name' as seen from group g: defined there or in a parent group."""
    while True:
        dims = dims_in(h5, g)
        if name in dims:
            return dims[name], g
        if g.name == "/":
            return None, None
        g = g.parent


def _new_dim(h5, f, g, name: str, length: int, unlimited: bool):
    ds = g.create_dataset(name, shape=(0 if unlimited else length,), maxshape=(None,) if unlimited else None,
                          dtype=">f4", chunks=(1,) if unlimited else None, track_order=True)
    ds.make_scale(f"{DIM_TAG}{0 if unlimited else length:10d}")
    ds.attrs.create("_Netcdf4Dimid", np.int32(_next_dimid(h5, f)))
    return ds


def _put_text_attr(h5, obj, name: str, value: str) -> None:
    raw = value.encode("utf-8")
    if name in obj.attrs:
        del obj.attrs[name]
    if not raw:
        obj.attrs.create(name, h5.Empty("S1"))
        return
    obj.attrs.create(name, np.bytes_(raw))


# -- values ----------------------------------------------------------------------------------------

def matlab_class(h5, dtype) -> str:
    if h5.check_string_dtype(dtype) is not None:
        info = h5.check_string_dtype(dtype)
        return "char" if info.length == 1 else "string"
    if dtype.kind == "S":
        return "char"
    if dtype.kind == "O":
        return "string"
    native = np.dtype(dtype).newbyteorder("=")
    for name, (dt, _) in TYPES.items():
        if dt == native:
            return name
    raise MatlabError(f"NetCDF data of type {dtype} is not supported.")


def default_fill(cls: str) -> Any:
    if cls == "string":
        return ""
    return TYPES[cls][1]


def fill_value(h5, ds) -> Any:
    if "_FillValue" in ds.attrs:
        v = ds.attrs["_FillValue"]
        return v if isinstance(v, (bytes, str)) else np.asarray(v).reshape(-1)[0]
    return default_fill(matlab_class(h5, ds.dtype))


def attr_out(h5, obj, name: str) -> Any:
    """An attribute as a MATLAB value: text as char, numbers as a row vector, NC_STRING as strings."""
    try:
        raw = obj.attrs[name]
    except (OSError, TypeError, ValueError):
        return np.zeros((0, 0))
    dtype = obj.attrs.get_id(name).dtype
    if isinstance(raw, h5.Empty):
        return char("") if dtype.kind == "S" else np.zeros((1, 0))
    info = h5.check_string_dtype(dtype)
    if info is not None and info.length is None:  # NC_STRING
        items = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
                 for x in np.atleast_1d(np.asarray(raw, dtype=object)).reshape(-1)]
        if len(items) == 1:
            return MString(items[0])
        data = np.empty((1, len(items)), dtype=object)
        data[0, :] = items
        return StringArray(data)
    if isinstance(raw, bytes):
        return char(raw.decode("utf-8", errors="replace").rstrip("\0"))
    if isinstance(raw, str):
        return char(raw)
    a = np.atleast_1d(np.asarray(raw))
    if a.dtype.kind == "S":
        return char(b"".join(a.reshape(-1).tolist()).decode("utf-8", errors="replace").rstrip("\0"))
    if a.dtype.kind not in "biuf":
        return np.zeros((0, 0))
    return np.ascontiguousarray(a.astype(a.dtype.newbyteorder("="))).reshape(1, -1)


def attr_write(h5, obj, name: str, v: Any, classic: bool, dtype=None) -> None:
    """Create or replace an attribute from a MATLAB value."""
    if isinstance(v, StringArray) or (isinstance(v, Cell) and all(is_text(x) for x in v.data.flat)):
        items = [("" if x is None else x) for x in v.data.flatten(order="F")] if isinstance(v, StringArray) \
            else [text(x) for x in v.data.flatten(order="F")]
        if classic:
            raise MatlabError("A file in the netcdf4_classic format cannot hold string array attributes; use "
                              "a character vector.")
        if name in obj.attrs:
            del obj.attrs[name]
        obj.attrs.create(name, np.array(items, dtype=object), dtype=h5.string_dtype())
        return
    if isinstance(v, MString) and not classic:
        if name in obj.attrs:
            del obj.attrs[name]
        obj.attrs.create(name, np.array([str(v)], dtype=object), dtype=h5.string_dtype())
        return
    if isinstance(v, MString) or (isinstance(v, np.ndarray) and v.dtype.kind == "U"):
        _put_text_attr(h5, obj, name, text(v, "the attribute value"))
        return
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        cls = class_name(v)
        if classic and cls not in CLASSIC_TYPES + ("logical",):
            raise MatlabError(f"Attributes of class '{cls}' cannot be stored in a netcdf4_classic file; use "
                              "double, single, int32, int16, int8 or text.")
        a = v.reshape(-1, order="F")
        a = a.astype(np.int8) if a.dtype == bool else a
        if dtype is not None:
            a = a.astype(dtype)
        if name in obj.attrs:
            del obj.attrs[name]
        obj.attrs.create(name, a)
        return
    raise MatlabError("A NetCDF attribute must be numeric or text.")


def _attr_struct(h5, obj) -> Any:
    names = [n for n in obj.attrs.keys() if n not in _HIDDEN]
    if not names:
        return np.zeros((0, 0))
    data = np.empty((1, len(names)), dtype=object)
    for k, n in enumerate(names):
        data[0, k] = {"Name": char(n), "Value": attr_out(h5, obj, n)}
    return Struct(data, ["Name", "Value"])


def to_matlab(h5, raw: np.ndarray, cls: str, nd: int) -> Any:
    """File-order data -> MATLAB array (dimensions reversed; one dimension gives a column)."""
    a = np.asarray(raw)
    if nd == 0:
        a = a.reshape(1, 1)
    elif nd == 1:
        a = a.reshape(-1, 1)
    else:
        a = a.transpose()
    if cls == "char":
        if a.dtype.kind == "O":
            codes = np.array([(x[:1] if isinstance(x, bytes) else str(x).encode("latin-1", "replace")[:1]) or b"\0"
                              for x in a.reshape(-1)], dtype="S1").reshape(a.shape)
            a = codes
        b = np.frombuffer(np.ascontiguousarray(a).astype("S1").tobytes(), dtype=np.uint8).reshape(a.shape)
        return np.ascontiguousarray(b.astype("<u4")).view("<U1").reshape(a.shape)
    if cls == "string":
        data = np.empty(a.shape, dtype=object)
        for idx in np.ndindex(a.shape):
            x = a[idx]
            data[idx] = x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x)
        return StringArray(data) if data.size != 1 else MString(data.flat[0])
    return np.ascontiguousarray(a.astype(a.dtype.newbyteorder("=")))


# -- the variables a location names ----------------------------------------------------------------

def find_var(h5, f, loc: str, fn: str):
    gpath, name = split_location(loc)
    g = find_group(h5, f, gpath, fn)
    vs = vars_in(h5, g)
    if name not in vs:
        raise MatlabError(f"The NetCDF file has no variable named '{loc}'.", "MATLAB:imagesci:netcdf:unknownLocation")
    return vs[name]


def read_var(h5, f, ds, start: list, count: list, stride: list, name: str):
    """Data of the block (file order) of a variable; past the stored extent of an unlimited dimension
    (up to the dimension's length) the fill value is returned."""
    dims = var_dims(h5, f, ds)
    nd = len(dims)
    sizes = [d[2] for d in dims]
    cls = matlab_class(h5, ds.dtype)
    if nd == 0:
        return np.asarray(ds[()]), cls
    counts, sl = [], []
    for k in range(nd):
        n = sizes[k]
        if start[k] >= n and n:
            raise MatlabError(f"ncread: start ({start[k] + 1}) is beyond the length ({n}) of dimension "
                              f"{nd - k} of '{name}'.", _ERR)
        c = len(range(start[k], n, stride[k])) if count[k] is None else count[k]
        stop = start[k] + (c - 1) * stride[k] + 1 if c else start[k]
        if stop > n:
            raise MatlabError(f"ncread: start, count and stride reach beyond the length ({n}) of dimension "
                              f"{nd - k} of '{name}'.", _ERR)
        counts.append(c)
        sl.append((start[k], stop, stride[k]))
    stored = list(ds.shape)
    if all(stop <= stored[k] for k, (_, stop, _) in enumerate(sl)):
        if 0 in counts:
            return np.zeros(counts, dtype=ds.dtype), cls
        return np.asarray(ds[tuple(slice(*s) for s in sl)]), cls
    out = np.empty(counts, dtype=ds.dtype if cls != "string" else object)
    out[...] = fill_value(h5, ds)
    have = [len(range(s[0], min(s[1], stored[k]), s[2])) if s[0] < stored[k] else 0 for k, s in enumerate(sl)]
    if all(have):
        out[tuple(slice(0, n) for n in have)] = ds[tuple(slice(s[0], s[0] + (n - 1) * s[2] + 1, s[2])
                                                      for s, n in zip(sl, have))]
    return out, cls


def unpack(h5, ds, out: Any, cls: str) -> Any:
    """NaN for the fill value of floating-point data, and scale_factor / add_offset applied."""
    if cls in ("char", "string") or not isinstance(out, np.ndarray):
        return out
    scaled = "add_offset" in ds.attrs or "scale_factor" in ds.attrs
    if out.dtype.kind == "f" or scaled:
        fill = fill_value(h5, ds)
        is_fill = out == fill
        if scaled:
            out = out.astype(float)
        if is_fill.any():
            out = out.astype(float if not scaled and out.dtype != np.float32 else out.dtype, copy=True)
            out[is_fill] = np.nan
        if scaled:
            out = out * float(np.asarray(ds.attrs.get("scale_factor", 1)).reshape(-1)[0]) + \
                float(np.asarray(ds.attrs.get("add_offset", 0)).reshape(-1)[0])
    return out


def stored_values(h5, ds, data: Any, name: str) -> np.ndarray:
    """MATLAB data converted to what the variable stores (NaN -> fill value, packing undone)."""
    cls = matlab_class(h5, ds.dtype)
    if cls == "string":
        if isinstance(data, MString):
            items = np.array([[str(data)]], dtype=object)
        elif isinstance(data, StringArray):
            items = np.empty(data.data.shape, dtype=object)
            for idx in np.ndindex(data.data.shape):
                items[idx] = "" if data.data[idx] is None else data.data[idx]
        elif isinstance(data, Cell) and all(is_text(x) for x in data.data.flat):
            items = np.empty(data.data.shape, dtype=object)
            for idx in np.ndindex(data.data.shape):
                items[idx] = text(data.data[idx])
        elif isinstance(data, np.ndarray) and data.dtype.kind == "U":
            items = np.array([["".join(data.reshape(-1).tolist())]], dtype=object)
        else:
            raise MatlabError(f"ncwrite: the variable '{name}' holds strings, so the data must be text.")
        return items
    if isinstance(data, MString):
        data = char(str(data))
    if not isinstance(data, np.ndarray):
        raise MatlabError("ncwrite: the data must be a numeric or char array.")
    if cls == "char":
        if data.dtype.kind != "U":
            raise MatlabError(f"ncwrite: the variable '{name}' holds characters, so the data must be text.")
        from ..values import char_codes
        return char_codes(data).astype(np.uint8).view("S1").reshape(data.shape)
    if data.dtype.kind == "U":
        raise MatlabError(f"ncwrite: the variable '{name}' is numeric, so the data must not be text.")
    if data.dtype.kind == "c":
        raise MatlabError("ncwrite: complex data cannot be stored in a NetCDF file.")
    dtype = np.dtype(ds.dtype).newbyteorder("=")
    if data.dtype.kind in "iu" and dtype.kind in "iu" and "scale_factor" not in ds.attrs \
            and "add_offset" not in ds.attrs:
        from ..values import saturate_int
        return saturate_int(data, dtype) if data.dtype != dtype else data
    vals = data.astype(float)
    if "add_offset" in ds.attrs or "scale_factor" in ds.attrs:
        vals = (vals - float(np.asarray(ds.attrs.get("add_offset", 0)).reshape(-1)[0])) / \
            float(np.asarray(ds.attrs.get("scale_factor", 1)).reshape(-1)[0])
        if dtype.kind in "iu":
            vals = np.round(vals)
    missing = np.isnan(vals)
    if dtype.kind in "iu":
        from ..values import saturate_int
        out = saturate_int(vals, dtype)
    else:
        out = vals.astype(dtype)
    if missing.any():
        out = out.copy()
        out[missing] = fill_value(h5, ds)
    return out


def write_var(h5, f, ds, data: Any, start: list, stride: list, name: str) -> None:
    dims = var_dims(h5, f, ds)
    nd = len(dims)
    values = stored_values(h5, ds, data, name)
    if nd == 0:
        if values.size != 1:
            raise MatlabError(f"ncwrite: the variable '{name}' holds one value.")
        ds[()] = values.reshape(())
        return
    shape = values.shape
    if nd == 1:
        if len(shape) > 2 or (len(shape) == 2 and min(shape) > 1):
            raise MatlabError(f"ncwrite: the variable '{name}' has one dimension, so the data must be a vector.")
        block = values.reshape(-1, order="F")
    else:
        if len(shape) > nd and any(n != 1 for n in shape[nd:]):
            raise MatlabError(f"ncwrite: the data has more dimensions than the variable '{name}' ({nd}).")
        shape = shape[:nd] if len(shape) > nd else shape
        block = values.reshape(tuple(shape) + (1,) * (nd - len(shape))).transpose()
    fstart, fstride = start, stride
    ends = [fstart[k] + (block.shape[k] - 1) * fstride[k] + 1 if block.shape[k] else fstart[k] for k in range(nd)]
    grow = list(ds.shape)
    for k, (dname, _, length, unlimited) in enumerate(dims):
        if not unlimited and ends[k] > length:
            raise MatlabError(f"ncwrite: the data does not fit in the variable '{name}' (dimension '{dname}' has "
                              f"length {length}).", _ERR)
        grow[k] = max(grow[k], ends[k])
    if tuple(grow) != tuple(ds.shape):
        try:
            ds.resize(tuple(grow))
        except (TypeError, ValueError) as e:
            raise MatlabError(f"ncwrite: cannot extend '{name}': {e}") from None
    if block.size:
        ds[tuple(slice(fstart[k], ends[k], fstride[k]) for k in range(nd))] = block


# -- creating variables ---------------------------------------------------------------------------

def _require_group(h5, f, path: str, classic: bool):
    g = f
    for part in [p for p in path.strip("/").split("/") if p]:
        if part not in g:
            if classic:
                raise MatlabError("nccreate: a file in the netcdf4_classic format cannot have groups; use "
                                  "'Format', 'netcdf4'.")
            g = g.create_group(part, track_order=True)
        else:
            g = g[part]
            if not isinstance(g, h5.Group):
                raise MatlabError(f"nccreate: '{g.name}' is a variable, not a group.")
    return g


def create_var(h5, f, loc: str, dim_spec: list, kind: str, opts: dict) -> None:
    """nccreate on a NetCDF-4 file. ``dim_spec`` is [(name, length or None for a given-by-name dim,
    unlimited)] in MATLAB order; ``opts`` holds FillValue, ChunkSize, DeflateLevel and Shuffle."""
    classic = file_format(f) == "netcdf4_classic"
    gpath, name = split_location(loc)
    if not name:
        raise MatlabError("nccreate: the variable needs a name.")
    if kind not in TYPES and kind != "string":
        raise MatlabError(f"nccreate: unknown Datatype '{kind}'. Types are: {', '.join(TYPES)}, string.")
    if classic and kind not in CLASSIC_TYPES:
        raise MatlabError(f"nccreate: the datatype '{kind}' cannot be stored in a netcdf4_classic file. Use "
                          "double, single, int32, int16, int8 or char, or 'Format', 'netcdf4'.")
    g = _require_group(h5, f, gpath, classic)
    if name in vars_in(h5, g):
        raise MatlabError(f"The variable '{loc}' already exists.", "MATLAB:imagesci:netcdf:variableExists")
    if name in g and not (isinstance(g[name], h5.Dataset) and _is_pure_dim(g[name])):
        raise MatlabError(f"nccreate: '{g[name].name}' already exists and is not a variable.")
    # resolve the dimensions (MATLAB order in, file order out)
    scales: list = []
    own_coordinate = False
    for dname, length, unlimited in dim_spec:
        scale, where = _visible_dim(h5, g, dname)
        if scale is not None:
            have = dim_length(h5, f, scale)
            have_unlimited = scale.maxshape[0] is None
            if length is None or (unlimited and have_unlimited) or (not unlimited and not have_unlimited
                                                                    and have == length):
                scales.append((dname, scale))
                continue
            if where is g:
                raise MatlabError(f"nccreate: the dimension '{dname}' already exists with another length.")
        elif length is None and not unlimited:
            raise MatlabError(f"nccreate: the dimension '{dname}' does not exist yet; give its length.")
        if dname == name and len(dim_spec) == 1:
            own_coordinate = True  # the variable is the dimension's coordinate variable
            scales.append((dname, None, length, unlimited))
            continue
        scales.append((dname, _new_dim(h5, f, g, dname, length or 0, unlimited)))
    if len({s[0] for s in scales}) != len(scales):
        raise MatlabError("nccreate: a variable cannot use the same dimension twice.")
    if classic:
        flags = [s[1].maxshape[0] is None if s[1] is not None else s[3] for s in scales]
        if sum(flags) > 1 or any(flags[:-1]):
            raise MatlabError("nccreate: in the netcdf4_classic format only the last dimension of a variable can "
                              "be unlimited (Inf).")
    file_scales = scales[::-1]
    shape, maxshape = [], []
    for s in file_scales:
        if s[1] is None:  # the coordinate variable of a new dimension
            shape.append(0 if s[3] else s[2])
            maxshape.append(None if s[3] else s[2])
        else:
            unl = s[1].maxshape[0] is None
            shape.append(0 if unl else dim_length(h5, f, s[1]))
            maxshape.append(None if unl else shape[-1])
    replace = None
    if name in g:  # a dimension without a variable becomes this coordinate variable
        replace = g[name]
        if len(scales) != 1 or scales[0][0] != name:
            raise MatlabError(f"nccreate: '{name}' is a dimension, so a variable of that name must be its "
                              "coordinate variable (one dimension, the same name).")
        unl = replace.maxshape[0] is None
        shape, maxshape = [0 if unl else int(replace.shape[0])], [None if unl else int(replace.shape[0])]
    dtype = h5.string_dtype() if kind == "string" else TYPES[kind][0]
    kw: dict = {"track_order": True}
    chunks = None
    if "ChunkSize" in opts:
        chunks = tuple(int(x) for x in np.asarray(opts["ChunkSize"], dtype=float).reshape(-1))[::-1]
        if len(chunks) != len(shape) or any(c < 1 for c in chunks):
            raise MatlabError("nccreate: ChunkSize needs one positive value per dimension.")
    if "DeflateLevel" in opts:
        level = int(np.asarray(opts["DeflateLevel"], dtype=float).reshape(-1)[0])
        if not 0 <= level <= 9:
            raise MatlabError("nccreate: DeflateLevel must be between 0 and 9.")
        if level:
            kw.update(compression="gzip", compression_opts=level)
    if "Shuffle" in opts and bool(np.asarray(opts["Shuffle"]).any()):
        kw["shuffle"] = True
    if any(m is None for m in maxshape) or (kw.keys() - {"track_order"}):
        chunks = chunks or tuple(max(1, min(n, 1024)) if m is not None else 1 for n, m in zip(shape, maxshape)) \
            or None
    if not shape:
        chunks = None
        kw = {"track_order": True}
    fill_attr = None
    if kind != "string":
        fill: Any = TYPES[kind][1]
        if "FillValue" in opts:
            fv = opts["FillValue"]
            if is_text(fv) and text(fv).lower() == "disable":
                fill = None
                kw["fill_time"] = "never"
            elif kind == "char":
                fill = text(fv, "the fill value").encode("latin-1", "replace")[:1] or b"\0"
                fill_attr = np.array(fill, dtype="S1")
            else:
                fill = np.asarray(fv, dtype=float).reshape(-1)[0]
                fill_attr = np.array([fill]).astype(dtype)
                fill = fill_attr[0]
        if fill is not None:
            kw["fillvalue"] = np.array(fill, dtype=dtype)[()]
    refs = []
    dimid = None
    if replace is not None:
        refs = [(f[r], int(k)) for r, k in _references(replace)]
        for other, k in refs:
            other.dims[k].detach_scale(replace)
        dimid = np.asarray(replace.attrs["_Netcdf4Dimid"]).flat[0]
        del g[name]
    try:
        ds = g.create_dataset(name, shape=tuple(shape), maxshape=tuple(maxshape) if shape else None, dtype=dtype,
                              chunks=chunks, **kw)
    except (ValueError, TypeError) as e:
        raise MatlabError(f"nccreate: cannot create '{loc}': {e}") from None
    if fill_attr is not None:
        ds.attrs.create("_FillValue", fill_attr)
    if replace is not None or own_coordinate:
        ds.make_scale(name)
        ds.attrs.create("_Netcdf4Dimid", np.int32(dimid if dimid is not None else _next_dimid(h5, f)))
        for other, k in refs:
            other.dims[k].attach_scale(ds)
        ids = [int(np.asarray(ds.attrs["_Netcdf4Dimid"]).flat[0])]
    else:
        ids = []
        for k, s in enumerate(file_scales):
            ds.dims[k].attach_scale(s[1])
            ids.append(int(np.asarray(s[1].attrs["_Netcdf4Dimid"]).flat[0]))
    if ids:
        ds.attrs.create("_Netcdf4Coordinates", np.array(ids, dtype=np.int32))


# -- describing a file ----------------------------------------------------------------------------

def _dim_struct(items: list) -> Any:
    """items: [(name, length, unlimited)]."""
    if not items:
        return np.zeros((0, 0))
    data = np.empty((1, len(items)), dtype=object)
    for k, (n, length, unl) in enumerate(items):
        data[0, k] = {"Name": char(n), "Length": np.array([[float(length)]]), "Unlimited": bool_scalar(unl)}
    return Struct(data, ["Name", "Length", "Unlimited"])


def var_info(h5, f, ds, path: str) -> dict:
    dims = var_dims(h5, f, ds)[::-1]  # MATLAB order
    cls = matlab_class(h5, ds.dtype)
    if "_FillValue" in ds.attrs:
        fill = attr_out(h5, ds, "_FillValue")
    elif cls == "char":
        fill = char("\0")
    elif cls == "string":
        fill = MString("")
    else:
        fill = np.array([[default_fill(cls)]], dtype=TYPES[cls][0])
    try:
        fill_time = ds.id.get_create_plist().get_fill_time()
        if fill_time == h5.h5d.FILL_TIME_NEVER:
            fill = char("disable")
    except (AttributeError, OSError):
        pass
    deflate = ds.compression_opts if ds.compression == "gzip" else None
    return {"Filename": char(os.path.abspath(path)), "Name": char(_base(ds.name)),
            "Dimensions": _dim_struct([(n, length, unl) for n, _, length, unl in dims]),
            "Size": np.array([[float(d[2]) for d in dims]]) if dims else np.array([[1.0]]),
            "Datatype": char(cls), "Attributes": _attr_struct(h5, ds),
            "ChunkSize": np.array([[float(c) for c in ds.chunks[::-1]]]) if ds.chunks else np.zeros((0, 0)),
            "FillValue": fill,
            "DeflateLevel": np.array([[float(deflate)]]) if deflate is not None else np.zeros((0, 0)),
            "Shuffle": bool_scalar(bool(ds.shuffle)), "Format": char(file_format(f))}


_VAR_FIELDS = ["Name", "Dimensions", "Size", "Datatype", "Attributes", "ChunkSize", "FillValue", "DeflateLevel",
               "Shuffle"]
_GROUP_FIELDS = ["Name", "Dimensions", "Variables", "Attributes", "Groups", "Format"]


def group_info(h5, f, g, path: str) -> dict:
    dims = [(n, dim_length(h5, f, s), s.maxshape[0] is None) for n, s in dims_in(h5, g).items()]
    vs = vars_in(h5, g)
    vdata = np.empty((1, len(vs)), dtype=object)
    for k, ds in enumerate(vs.values()):
        info = var_info(h5, f, ds, path)
        vdata[0, k] = {x: info[x] for x in _VAR_FIELDS}
    subs = groups_in(h5, g)
    gdata = np.empty((1, len(subs)), dtype=object)
    for k, sub in enumerate(subs):
        info = group_info(h5, f, sub, path)
        gdata[0, k] = {x: info[x] for x in _GROUP_FIELDS}
    return {"Filename": char(os.path.abspath(path)), "Name": char(g.name if g.name == "/" else _base(g.name)),
            "Dimensions": _dim_struct(dims), "Variables": Struct(vdata, _VAR_FIELDS) if vs else np.zeros((0, 0)),
            "Attributes": _attr_struct(h5, g), "Groups": Struct(gdata, _GROUP_FIELDS) if subs else np.zeros((0, 0)),
            "Format": char(file_format(f))}


def _show(v: Any) -> str:
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        return "'" + "".join(v.reshape(-1).tolist()) + "'"
    if isinstance(v, MString):
        return f'"{v}"'
    if isinstance(v, StringArray):
        return " ".join(f'"{x or ""}"' for x in v.data.reshape(-1))
    flat = np.asarray(v).reshape(-1)
    return " ".join(f"{x:g}" for x in flat)


def disp_lines(h5, f, g, only: str | None, indent: str) -> list[str]:
    pad = indent + "           "
    lines: list[str] = []
    attrs = [n for n in g.attrs.keys() if n not in _HIDDEN]
    if attrs and only is None:
        lines.append(f"{indent}{'Global ' if g.name == '/' else ''}Attributes:")
        w = max(len(k) for k in attrs)
        lines += [f"{pad}{k:<{w}} = {_show(attr_out(h5, g, k))}" for k in attrs]
    dims = dims_in(h5, g)
    if dims and only is None:
        lines.append(f"{indent}Dimensions:")
        w = max(len(k) for k in dims)
        for k, s in dims.items():
            extra = "  (UNLIMITED)" if s.maxshape[0] is None else ""
            lines.append(f"{pad}{k:<{w}} = {dim_length(h5, f, s)}{extra}")
    vs = vars_in(h5, g)
    if only is not None:
        vs = {only: vs[only]}
    if vs:
        lines.append(f"{indent}Variables:")
    for name, ds in vs.items():
        dims_m = var_dims(h5, f, ds)[::-1]
        size = "x".join(str(d[2]) for d in dims_m) or "1x1"
        lines += [f"{indent}    {name}", f"{pad}Size:       {size}",
                  f"{pad}Dimensions: {','.join(d[0] for d in dims_m)}",
                  f"{pad}Datatype:   {matlab_class(h5, ds.dtype)}"]
        vattrs = [n for n in ds.attrs.keys() if n not in _HIDDEN]
        if vattrs:
            lines.append(f"{pad}Attributes:")
            w = max(len(k) for k in vattrs)
            lines += [f"{pad}            {k:<{w}} = {_show(attr_out(h5, ds, k))}" for k in vattrs]
    subs = groups_in(h5, g) if only is None else []
    if subs:
        lines.append(f"{indent}Groups:")
        for sub in subs:
            lines.append(f"{indent}    {sub.name}")
            lines += disp_lines(h5, f, sub, None, indent + "    ")
    return lines
