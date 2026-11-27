"""Loading and calling MEX files, and the mx/mex API they call back into.

Design (see docs/MEX.md for the user-facing version):

* **Arrays.** An ``mxArray*`` the C code sees is an opaque handle: a number
  that keys :attr:`MexRuntime.arrays`.  The array itself is an
  :class:`MxArray` whose numeric data lives in a NumPy buffer in column-major
  order, so ``mxGetPr`` and friends return the address of that buffer and C
  reads and writes it directly.  Complex data is kept interleaved
  (re, im, re, im, ...) and split into two buffers only when the separate
  API (``mxGetPi``) asks for it.  Cells and structs hold the handles of their
  elements, which they own.

* **Lifetime.** Every call of a MEX file opens a scope.  Arrays and
  ``mxMalloc`` blocks created while it runs belong to the scope and are
  released when the call returns (after the outputs were converted to mylab
  values), unless ``mexMakeArrayPersistent``/``mexMakeMemoryPersistent`` was
  used; persistent ones live until the MEX file is cleared.

* **Errors.** A Python exception must never cross a C frame.  Every callback
  catches everything, stores the exception in :attr:`MexRuntime.pending`,
  sets the table's ``pending`` flag and returns a harmless value.  The inline
  wrapper in the header sees the flag and unwinds (longjmp in C, a C++
  exception in C++) to the entry point, which returns 1; the exception is
  then raised here, in Python, where try/catch in mylab code can handle it.
"""

from __future__ import annotations

import atexit
import ctypes
import fnmatch
import itertools
import os
import re
import shutil
import sys
import tempfile
import weakref
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, Struct, as_2d, class_name
from . import IN_BROWSER, MEX_SUFFIX, spec

# mxClassID values (the order of the enum in include/matrix.h)
(UNKNOWN, CELL, STRUCT, LOGICAL, CHAR, VOID, DOUBLE, SINGLE, INT8, UINT8, INT16, UINT16, INT32, UINT32,
 INT64, UINT64, FUNCTION, OPAQUE, OBJECT) = range(19)

DTYPES = {
    DOUBLE: np.dtype(np.float64), SINGLE: np.dtype(np.float32),
    INT8: np.dtype(np.int8), UINT8: np.dtype(np.uint8), INT16: np.dtype(np.int16), UINT16: np.dtype(np.uint16),
    INT32: np.dtype(np.int32), UINT32: np.dtype(np.uint32), INT64: np.dtype(np.int64), UINT64: np.dtype(np.uint64),
    LOGICAL: np.dtype(np.bool_), CHAR: np.dtype(np.uint16),
}
NUMERIC = frozenset(range(DOUBLE, UINT64 + 1))
CLASS_NAMES = {
    UNKNOWN: "unknown", CELL: "cell", STRUCT: "struct", LOGICAL: "logical", CHAR: "char", VOID: "void",
    DOUBLE: "double", SINGLE: "single", INT8: "int8", UINT8: "uint8", INT16: "int16", UINT16: "uint16",
    INT32: "int32", UINT32: "uint32", INT64: "int64", UINT64: "uint64", FUNCTION: "function_handle",
    OPAQUE: "opaque", OBJECT: "object",
}
_REAL_OF = {np.dtype(np.complex128): DOUBLE, np.dtype(np.complex64): SINGLE}
_CLASS_OF = {dt: cls for cls, dt in DTYPES.items() if cls != CHAR}
INDEX = np.dtype(np.uintp)  # mwIndex / mwSize
_FIELD_NAME = re.compile(r"[A-Za-z]\w{0,62}\Z")
_HANDLE_STEP = 16


class MxArray:
    """The Python side of one mxArray (see the module docstring)."""

    __slots__ = ("cls", "dims", "dims_buf", "cplx", "sparse", "data", "imag", "ir", "jc", "cells", "fields",
                 "fvals", "value", "parent", "persistent", "owner")

    def __init__(self, cls: int, dims):
        self.cls = cls
        self.cplx = False
        self.sparse = False
        self.data: np.ndarray | None = None   # numeric data; interleaved when complex and imag is None
        self.imag: np.ndarray | None = None   # imaginary part while split for the separate-complex API
        self.ir: np.ndarray | None = None
        self.jc: np.ndarray | None = None
        self.cells: list[int] | None = None
        self.fields: list[str] | None = None
        self.fvals: list[list[int]] | None = None
        self.value: Any = None                # a mylab value C cannot look inside (handles, objects, strings)
        self.parent = 0
        self.persistent = False
        self.owner = None
        self.set_dims(dims)

    def set_dims(self, dims) -> None:
        dims = [int(d) for d in dims]
        if not dims:
            dims = [0, 0]
        elif len(dims) == 1:
            dims.append(1)
        while len(dims) > 2 and dims[-1] == 1:
            dims.pop()
        self.dims = tuple(dims)
        self.dims_buf = np.array(dims, dtype=INDEX)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    def children(self) -> list[int]:
        if self.cells is not None:
            return [h for h in self.cells if h]
        if self.fvals is not None:
            return [h for row in self.fvals for h in row if h]
        return []


class _Block:
    __slots__ = ("buf", "persistent", "owner")

    def __init__(self, buf: np.ndarray, owner):
        self.buf = buf
        self.persistent = False
        self.owner = owner


class _Scope:
    __slots__ = ("arrays", "mem")

    def __init__(self):
        self.arrays: set[int] = set()
        self.mem: set[int] = set()


class MexLibrary:
    """One loaded MEX file."""

    def __init__(self, name: str, path: str, cdll, copy: str | None):
        self.name = name
        self.path = path
        self.cdll = cdll
        self.copy = copy          # the private copy that was loaded (kept only where it cannot be deleted)
        self.locks = 0
        self.trap = False
        self.entry = cdll.mylab_mex_entry
        self.entry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.entry.restype = ctypes.c_int
        self.run_atexit = cdll.mylab_mex_run_atexit
        self.run_atexit.argtypes = [ctypes.c_void_p]
        self.run_atexit.restype = ctypes.c_int
        self.has_atexit = cdll.mylab_mex_has_atexit
        self.has_atexit.restype = ctypes.c_int


# -- the ctypes view of the table ------------------------------------------------------------------

def _ctype(t: str, ret: bool):
    if t == "void":
        return None
    if t.endswith("*"):
        return ctypes.c_char_p if (t == "const char*" and not ret) else ctypes.c_void_p
    return {"int": ctypes.c_int, "mxClassID": ctypes.c_int, "mxComplexity": ctypes.c_int,
            "double": ctypes.c_double, "bool": ctypes.c_bool, "size_t": ctypes.c_size_t,
            "mwSize": ctypes.c_size_t, "mwIndex": ctypes.c_size_t}[t]


_ENTRIES = spec.entries()
_PROTOS = {e.name: ctypes.CFUNCTYPE(_ctype(e.ret, True), *(_ctype(t, False) for t, _ in e.args))
           for e in _ENTRIES}


class _Table(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_int), ("pending", ctypes.c_int)] + \
               [(e.name, _PROTOS[e.name]) for e in _ENTRIES]


def _default(e: spec.Entry):
    if e.ret == "void" or e.ret.endswith("*"):
        return None
    if e.ret == "bool":
        return False
    if e.ret == "double":
        return 0.0
    return 0


def _converter(e: spec.Entry):
    if e.ret == "void":
        return lambda r: None
    if e.ret.endswith("*"):
        return lambda r: int(r) or None
    return {"bool": bool, "double": float}.get(e.ret, int)


def _text(b: bytes | None) -> str:
    return "" if b is None else b.decode("utf-8", "replace")


def _ptr(a: np.ndarray | None):
    if a is None or a.size == 0:
        return None
    return a.ctypes.data


def _sizes(ptr, n: int) -> list[int]:
    n = int(n)
    if n == 0:
        return []
    if not ptr:
        raise MatlabError("The dimensions pointer is NULL.", "mylab:mex:nullPointer")
    return list((ctypes.c_size_t * n).from_address(ptr))


def _strings(ptr, n: int) -> list[str]:
    n = int(n)
    if n <= 0:
        return []
    if not ptr:
        raise MatlabError("The list of strings is NULL.", "mylab:mex:nullPointer")
    return [_text(s) for s in (ctypes.c_char_p * n).from_address(ptr)]


def _shutdown(ref) -> None:
    rt = ref()
    if rt is not None:
        try:
            rt.clear(None, force=True)
        except Exception:  # noqa: BLE001 - Python is exiting: nothing useful to do with the error
            pass
        if rt._tmp:
            shutil.rmtree(rt._tmp, ignore_errors=True)


class MexRuntime:
    """MEX files loaded by one interpreter, and the arrays and memory they use."""

    def __init__(self, interp):
        self.interp = interp
        self.libs: dict[str, MexLibrary] = {}
        self.arrays: dict[int, MxArray] = {}
        self.mem: dict[int, _Block] = {}
        self.scopes: list[_Scope] = []
        self.stack: list[tuple[MexLibrary, Any]] = []   # (library, caller's workspace) per active call
        self.pending: BaseException | None = None
        self._ids = itertools.count(1 << 16)
        self._cstrings: dict[str, ctypes.Array] = {}
        self._tmp: str | None = None
        self._loads = itertools.count(1)
        self.table = _Table()
        self.table.abi_version = spec.ABI_VERSION
        self._callbacks = []
        for e in _ENTRIES:
            cb = _PROTOS[e.name](self._wrap(e))
            self._callbacks.append(cb)   # ctypes callbacks must stay alive as long as the table
            setattr(self.table, e.name, cb)
        self.table_ptr = ctypes.addressof(self.table)
        atexit.register(_shutdown, weakref.ref(self))

    # -- callbacks -------------------------------------------------------------------------------
    def _wrap(self, e: spec.Entry):
        method = getattr(self, e.name)
        default = _default(e)
        name = e.name
        conv = _converter(e)

        def callback(*args):
            try:
                r = method(*args)
                return default if r is None else conv(r)
            except BaseException as exc:  # noqa: BLE001 - nothing may propagate into C: see the module docstring
                self._fail(exc, name)
                return default
        return callback

    def _fail(self, exc: BaseException, name: str) -> None:
        if not isinstance(exc, (MatlabError, KeyboardInterrupt)):
            err = MatlabError(f"{name}: {type(exc).__name__}: {exc}", "mylab:mex:apiError")
            err.__cause__ = exc
            exc = err
        if self.pending is None:
            self.pending = exc
        self.table.pending = 1

    def _take_error(self) -> BaseException:
        err = self.pending or MatlabError("The MEX file stopped with an error.", "mylab:mex:error")
        self.pending = None
        self.table.pending = 0
        return err

    def cstr(self, s: str) -> int:
        """Address of a NUL-terminated copy of ``s`` that stays valid as long as the runtime."""
        buf = self._cstrings.get(s)
        if buf is None:
            buf = self._cstrings[s] = ctypes.create_string_buffer(s.encode("utf-8"))
        return ctypes.addressof(buf)

    # -- loading -------------------------------------------------------------------------------------
    def load(self, path: str) -> MexLibrary:
        real = os.path.realpath(path)
        lib = self.libs.get(real)
        if lib is not None:
            return lib
        name = os.path.basename(path)[: -len(MEX_SUFFIX)]
        try:
            with open(real, "rb") as f:
                blob = f.read()
        except OSError as e:
            raise MatlabError(f"Cannot read the MEX file '{path}': {e.strerror or e}.", "mylab:mex:load")
        if b"mylab_mex_entry" not in blob:
            raise MatlabError(
                f"'{path}' was not built by mylab's mex. MEX files built for MATLAB link to MATLAB's own "
                f"libraries and cannot run in mylab: rebuild it from its C/C++ source with mylab, "
                f"e.g. mex {name}.c", "mylab:mex:notMylabMex")
        if self._tmp is None:
            self._tmp = tempfile.mkdtemp(prefix="mylab-mex-")
        # Load a private copy: the original can then be rebuilt (or deleted) while it is loaded, and a
        # rebuilt file is a new library rather than the cached old one.
        copy = os.path.join(self._tmp, f"{name}-{next(self._loads)}{MEX_SUFFIX}")
        shutil.copyfile(real, copy)
        try:
            cdll = ctypes.CDLL(copy)
        except OSError as e:
            _remove(copy)
            raise MatlabError(f"Cannot load the MEX file '{path}': {e}", "mylab:mex:load")
        if os.name != "nt":
            _remove(copy)  # a loaded library stays usable after its file is deleted
            copy = None
        try:
            abi = cdll.mylab_mex_abi()
        except AttributeError:
            abi = -1
        if abi != spec.ABI_VERSION:
            raise MatlabError(f"'{path}' was built by a different version of mylab. Rebuild it with mex.",
                              "mylab:mex:abiMismatch")
        lib = self.libs[real] = MexLibrary(name, real, cdll, copy)
        return lib

    def loaded(self) -> list[MexLibrary]:
        return list(self.libs.values())

    def clear(self, patterns, force: bool = False) -> None:
        for lib in list(self.libs.values()):
            if patterns is not None and not any(fnmatch.fnmatchcase(lib.name, p) for p in patterns):
                continue
            if lib.locks and not force:
                continue
            self.unload(lib)

    def unload_path(self, path: str) -> None:
        """Unload the MEX file at ``path`` if it is loaded (mex does this before replacing it)."""
        lib = self.libs.get(os.path.realpath(path))
        if lib is None:
            return
        if lib.locks:
            raise MatlabError(f"The MEX file '{lib.name}' is locked (mexLock) and cannot be replaced.",
                              "mylab:mex:locked")
        self.unload(lib)

    def unload(self, lib: MexLibrary) -> None:
        if any(active is lib for active, _ in self.stack):
            return  # still running (a MEX file that clears itself through mexEvalString)
        try:
            if lib.has_atexit():
                self._enter(lib, None)
                try:
                    if lib.run_atexit(self.table_ptr):
                        err = self._take_error()
                        self.interp.out(f"Warning: the exit function of MEX file '{lib.name}' raised an error: "
                                        f"{getattr(err, 'message', err)}\n")
                finally:
                    self._leave()
        finally:
            self.libs.pop(lib.path, None)
            for h, a in list(self.arrays.items()):
                if a.persistent and a.owner is lib:
                    a.persistent = False
                    self._destroy(h)
            for addr, b in list(self.mem.items()):
                if b.persistent and b.owner is lib:
                    del self.mem[addr]
            _close(lib.cdll)
            if lib.copy:
                _remove(lib.copy)

    # -- calling -------------------------------------------------------------------------------------
    def _enter(self, lib: MexLibrary, caller) -> None:
        self.scopes.append(_Scope())
        self.stack.append((lib, caller))

    def _leave(self) -> None:
        self.stack.pop()
        scope = self.scopes.pop()
        for h in scope.arrays:
            a = self.arrays.get(h)
            if a is None:
                continue
            root, rh = a, h
            while root.parent:
                parent = self.arrays.get(root.parent)
                if parent is None:
                    break
                root, rh = parent, root.parent
            if root.persistent or (rh != h and rh not in scope.arrays and rh in self.arrays):
                continue  # kept alive by a persistent array, or by an array of an outer call
            self._destroy(rh)
        for addr in scope.mem:
            b = self.mem.get(addr)
            if b is not None and not b.persistent:
                del self.mem[addr]

    def call(self, path: str, args: list, nargout: int, caller) -> list:
        if IN_BROWSER:
            raise MatlabError("MEX files cannot run when mylab runs in the browser: they are native code. "
                              "Run the program with mylab installed on your computer.", "mylab:mex:browser")
        lib = self.load(path)
        self._enter(lib, caller)
        try:
            prhs_h = [self.to_mx(v) for v in args]
            prhs = (ctypes.c_void_p * max(len(prhs_h), 1))(*prhs_h)
            nslots = max(nargout, 1)
            plhs = (ctypes.c_void_p * nslots)()
            self.pending = None
            self.table.pending = 0
            status = lib.entry(self.table_ptr, nargout, plhs, len(prhs_h), prhs)
            if status or self.pending is not None:
                raise self._take_error()
            outs = []
            for i in range(nslots):
                h = plhs[i]
                if not h:
                    if i < nargout:
                        raise MatlabError(f"One or more output arguments were not assigned during the call to "
                                          f"MEX file '{lib.name}'.", "MATLAB:unassignedOutputs")
                    break
                outs.append(self.from_mx(h))
            return outs
        finally:
            self._leave()

    # -- arrays ----------------------------------------------------------------------------------------
    def _register(self, a: MxArray) -> int:
        h = next(self._ids) * _HANDLE_STEP
        self.arrays[h] = a
        if self.scopes:
            self.scopes[-1].arrays.add(h)
        return h

    def _orphan(self, h: int) -> None:
        """An array no container holds any more: freed when the current call ends."""
        a = self.arrays.get(h)
        if a is not None:
            a.parent = 0
            if self.scopes:
                self.scopes[-1].arrays.add(h)

    def _adopt_child(self, parent_h: int, h) -> int:
        if not h:
            return 0
        child = self._get(h)
        if child.parent and child.parent != parent_h:
            self._detach(h)
        child.parent = parent_h
        return h

    def _detach(self, h: int) -> None:
        a = self.arrays.get(h)
        if a is None or not a.parent:
            return
        p = self.arrays.get(a.parent)
        if p is not None:
            if p.cells is not None:
                p.cells = [0 if c == h else c for c in p.cells]
            elif p.fvals is not None:
                p.fvals = [[0 if c == h else c for c in row] for row in p.fvals]
        a.parent = 0

    def _destroy(self, h: int) -> None:
        a = self.arrays.pop(h, None)
        if a is None:
            return
        for c in a.children():
            child = self.arrays.get(c)
            if child is not None and child.parent == h:
                self._destroy(c)

    def _get(self, h) -> MxArray:
        a = self.arrays.get(h) if h else None
        if a is None:
            if not h:
                raise MatlabError("The mxArray pointer is NULL.", "mylab:mex:nullPointer")
            raise MatlabError("The mxArray pointer is not a valid array (was it destroyed, or freed after an "
                              "earlier call? Use mexMakeArrayPersistent to keep an array between calls).",
                              "mylab:mex:invalidArray")
        return a

    def _new(self, cls: int, dims, cplx: bool = False) -> int:
        a = MxArray(cls, dims)
        n = a.numel
        if cls in DTYPES:
            a.cplx = bool(cplx) and cls in NUMERIC
            a.data = np.zeros(n * (2 if a.cplx else 1), dtype=DTYPES[cls])
        elif cls == CELL:
            a.cells = [0] * n
        elif cls == STRUCT:
            a.fields = []
            a.fvals = [[] for _ in range(n)]
        return self._register(a)

    def _new_sparse(self, m: int, n: int, nzmax: int, cls: int, cplx: bool) -> int:
        a = MxArray(cls, (m, n))
        a.sparse = True
        a.cplx = bool(cplx) and cls == DOUBLE
        nz = max(int(nzmax), 1)
        a.ir = np.zeros(nz, dtype=INDEX)
        a.jc = np.zeros(int(n) + 1, dtype=INDEX)
        a.data = np.zeros(nz * (2 if a.cplx else 1), dtype=DTYPES[cls])
        return self._register(a)

    @staticmethod
    def _split(a: MxArray) -> None:
        if a.cplx and a.imag is None:
            d = a.data
            a.data = np.ascontiguousarray(d[0::2])
            a.imag = np.ascontiguousarray(d[1::2])

    @staticmethod
    def _interleave(a: MxArray) -> None:
        if a.cplx and a.imag is not None:
            n = max(a.data.size, a.imag.size)
            d = np.zeros(2 * n, dtype=a.data.dtype)
            d[0:2 * a.data.size:2] = a.data
            d[1:2 * a.imag.size:2] = a.imag
            a.data, a.imag = d, None

    def _set_struct_fields(self, h: int, names: list[str]) -> None:
        a = self.arrays[h]
        for nm in names:
            if not _FIELD_NAME.match(nm):
                raise MatlabError(f"'{nm}' is not a valid field name.", "MATLAB:InvalidFieldName")
            if nm in a.fields:
                raise MatlabError(f"Duplicate field name '{nm}'.", "MATLAB:DuplicateFieldName")
            a.fields.append(nm)
        a.fvals = [[0] * len(a.fields) for _ in range(a.numel)]

    # -- conversions ----------------------------------------------------------------------------------
    def to_mx(self, v: Any, parent: int = 0) -> int:
        """A new array (in the current scope) holding a copy of the mylab value ``v``."""
        if isinstance(v, np.ndarray) and v.dtype != object:
            dims = v.shape if v.ndim >= 2 else (1, v.size)
            kind = v.dtype.kind
            if kind == "U":
                flat = np.ascontiguousarray(v.ravel(order="F"))
                codes = flat.view(np.uint32) if flat.size else np.zeros(0, np.uint32)
                codes = np.where(codes > 0xFFFF, 0xFFFD, codes).astype(np.uint16)
                a = MxArray(CHAR, dims)
                a.data = codes
            elif kind == "c":
                cls = _REAL_OF.get(v.dtype, DOUBLE)
                flat = v.ravel(order="F")
                a = MxArray(cls, dims)
                a.cplx = True
                a.data = np.empty(2 * flat.size, dtype=DTYPES[cls])
                a.data[0::2] = flat.real
                a.data[1::2] = flat.imag
            else:
                cls = _CLASS_OF.get(v.dtype)
                if cls is None:
                    v = v.astype(np.float64)
                    cls = DOUBLE
                a = MxArray(cls, dims)
                a.data = np.array(v.ravel(order="F"), dtype=DTYPES[cls])
            a.parent = parent
            return self._register(a)
        if isinstance(v, Cell):
            a = MxArray(CELL, v.data.shape)
            a.parent = parent
            h = self._register(a)
            a.cells = [self.to_mx(x, h) for x in v.data.ravel(order="F")]
            return h
        if isinstance(v, Struct):
            a = MxArray(STRUCT, v.data.shape)
            a.parent = parent
            h = self._register(a)
            a.fields = list(v.fields)
            a.fvals = [[self.to_mx(elem[f], h) if f in elem else 0 for f in a.fields]
                       for elem in v.data.ravel(order="F")]
            return h
        sm = _as_sparse(v)
        if sm is not None:
            m = sm.m
            cls = LOGICAL if m.dtype == np.bool_ else DOUBLE
            h = self._new_sparse(m.shape[0], m.shape[1], m.nnz, cls, m.dtype.kind == "c")
            a = self.arrays[h]
            a.parent = parent
            nnz = m.nnz
            a.ir[:nnz] = m.indices
            a.jc[:] = m.indptr
            if a.cplx:
                a.data[0:2 * nnz:2] = m.data.real
                a.data[1:2 * nnz:2] = m.data.imag
            else:
                a.data[:nnz] = m.data
            return h
        # anything C cannot look inside: kept as it is, and passed back unchanged
        shape = getattr(v, "shape", (1, 1))
        try:
            dims = tuple(int(d) for d in shape)
        except (TypeError, ValueError):
            dims = (1, 1)
        a = MxArray(FUNCTION if isinstance(v, FunctionHandle) else OBJECT, dims)
        a.value = v
        a.parent = parent
        return self._register(a)

    def from_mx(self, h) -> Any:
        """The mylab value an array holds (a copy)."""
        a = self._get(h)
        if a.value is not None or a.cls in (FUNCTION, OBJECT, OPAQUE):
            return a.value
        n = a.numel
        if a.cls == CELL:
            flat = np.empty(n, dtype=object)
            for i in range(n):
                c = a.cells[i] if i < len(a.cells) else 0
                flat[i] = self.from_mx(c) if c else np.zeros((0, 0))
            return Cell(flat.reshape(a.dims, order="F"))
        if a.cls == STRUCT:
            flat = np.empty(n, dtype=object)
            for i in range(n):
                row = a.fvals[i] if i < len(a.fvals) else []
                flat[i] = {f: (self.from_mx(row[k]) if k < len(row) and row[k] else np.zeros((0, 0)))
                           for k, f in enumerate(a.fields)}
            return Struct(flat.reshape(a.dims, order="F"), list(a.fields))
        if a.sparse:
            return self._sparse_value(a)
        if a.cls not in DTYPES:
            raise MatlabError(f"An mxArray of class '{CLASS_NAMES.get(a.cls, a.cls)}' cannot be returned.",
                              "mylab:mex:badClass")
        if a.cplx:
            if a.cls not in (DOUBLE, SINGLE):
                raise MatlabError("mylab does not support complex integer arrays.", "mylab:mex:complexInteger")
            re_, im = (a.data, a.imag) if a.imag is not None else (a.data[0::2], a.data[1::2])
            if re_.size < n or im.size < n:
                raise _too_small()
            out = np.empty(n, dtype=np.complex128 if a.cls == DOUBLE else np.complex64)
            out.real = re_[:n]
            out.imag = im[:n]
        else:
            if a.data.size < n:
                raise _too_small()
            out = np.array(a.data[:n])
            if a.cls == LOGICAL:
                out = out.view(np.uint8) != 0
            elif a.cls == CHAR:
                out = out.astype(np.uint32).view("<U1") if n else np.zeros(0, dtype="<U1")
        return as_2d(out.reshape(a.dims, order="F"))

    def _sparse_value(self, a: MxArray):
        from scipy import sparse as sps

        from ..builtins.sparse import SparseMatrix
        m, n = a.dims[0], int(np.prod(a.dims[1:]))
        jc = a.jc[: n + 1].astype(np.int64)
        if jc.size != n + 1:
            raise MatlabError("Sparse array: Jc has fewer than N+1 entries.", "mylab:mex:badSparse")
        nnz = int(jc[-1])
        if nnz > a.ir.size or np.any(np.diff(jc) < 0) or jc[0] != 0:
            raise MatlabError("Sparse array: the column starts in Jc are not valid.", "mylab:mex:badSparse")
        ir = a.ir[:nnz].astype(np.int64)
        if nnz and (ir.max() >= m):
            raise MatlabError("Sparse array: a row index in Ir is out of range.", "mylab:mex:badSparse")
        if a.cplx:
            re_, im = (a.data, a.imag) if a.imag is not None else (a.data[0::2], a.data[1::2])
            vals = re_[:nnz] + 1j * im[:nnz]
        elif a.cls == LOGICAL:
            vals = a.data[:nnz].view(np.uint8) != 0
        else:
            vals = np.array(a.data[:nnz], dtype=np.float64)
        return SparseMatrix(sps.csc_matrix((vals, ir, jc), shape=(m, n)))

    def _copy(self, h: int, parent: int = 0) -> int:
        a = self._get(h)
        b = MxArray(a.cls, a.dims)
        b.cplx, b.sparse, b.value, b.parent = a.cplx, a.sparse, a.value, parent
        for attr in ("data", "imag", "ir", "jc"):
            x = getattr(a, attr)
            setattr(b, attr, None if x is None else x.copy())
        nh = self._register(b)
        if a.cells is not None:
            b.cells = [self._copy(c, nh) if c else 0 for c in a.cells]
        if a.fvals is not None:
            b.fields = list(a.fields)
            b.fvals = [[self._copy(c, nh) if c else 0 for c in row] for row in a.fvals]
        return nh

    def _resize_elements(self, a: MxArray, h: int) -> None:
        """Keep a cell's or struct's element list in step with new dimensions."""
        n = a.numel
        if a.cells is not None:
            for c in a.cells[n:]:
                if c:
                    self._orphan(c)
            a.cells = (a.cells + [0] * n)[:n]
        elif a.fvals is not None:
            for row in a.fvals[n:]:
                for c in row:
                    if c:
                        self._orphan(c)
            a.fvals = (a.fvals + [[0] * len(a.fields) for _ in range(n)])[:n]

    def _adopt_memory(self, ptr) -> np.ndarray:
        if not ptr:
            return np.zeros(0, dtype=np.uint8)
        b = self.mem.pop(ptr, None)
        if b is None:
            raise MatlabError("The data pointer did not come from mxMalloc, mxCalloc or mxRealloc (only that "
                              "memory can be handed to an mxArray).", "mylab:mex:foreignPointer")
        return b.buf

    @staticmethod
    def _view(buf: np.ndarray, dtype: np.dtype) -> np.ndarray:
        k = buf.size // dtype.itemsize
        return buf[: k * dtype.itemsize].view(dtype)

    def _lib(self) -> MexLibrary | None:
        return self.stack[-1][0] if self.stack else None

    def _workspace(self, ws: bytes | None):
        w = _text(ws).lower()
        if w == "base":
            return self.interp.base
        if w == "caller":
            caller = self.stack[-1][1] if self.stack else None
            return caller if caller is not None else self.interp.base
        if w == "global":
            return "global"
        raise MatlabError(f"The workspace must be 'base', 'caller' or 'global', not '{w}'.", "mylab:mex:workspace")

    # ================================================================================================
    # The functions of the table (names as in spec.TABLE).  Arguments arrive as ctypes passes them:
    # pointers as int or None, const char * as bytes or None.
    # ================================================================================================

    # -- internal ------------------------------------------------------------------------------------
    def mylab_print(self, text):
        self.interp.out(_text(text))

    def mylab_error(self, identifier, message):
        err = MatlabError(_text(message), _text(identifier))
        if self.pending is None:
            self.pending = err
        self.table.pending = 1

    def mylab_warning(self, identifier, message):
        from ..builtins.system import _issue_warning
        _issue_warning(self.interp, _text(message), _text(identifier))

    def mylab_data(self, h, which):
        a = self._get(h)
        if a.data is None:
            return None
        if which == spec.DATA_INTERLEAVED:
            self._interleave(a)
            return _ptr(a.data)
        if which == spec.DATA_IMAG:
            if not a.cplx:
                return None
            self._split(a)
            return _ptr(a.imag)
        self._split(a)
        return _ptr(a.data)

    def mylab_set_data(self, h, ptr, which):
        a = self._get(h)
        if a.cls not in DTYPES:
            raise MatlabError("Only numeric, logical and char arrays have data to set.", "mylab:mex:badClass")
        arr = self._view(self._adopt_memory(ptr), DTYPES[a.cls])
        if which == spec.DATA_INTERLEAVED:
            if a.cplx:
                a.imag = None
            a.data = arr
        elif which == spec.DATA_IMAG:
            if a.cls not in NUMERIC:
                raise MatlabError("Only numeric arrays can have an imaginary part.", "mylab:mex:badClass")
            if not a.cplx:
                a.cplx = True
                a.imag = arr
            else:
                self._split(a)
                a.imag = arr
        else:
            self._split(a)
            a.data = arr
        return 1

    def mylab_typed_data(self, h, cls, cplx):
        a = self._get(h)
        if a.cls != cls or a.cplx != bool(cplx) or a.data is None:
            return None
        self._interleave(a)
        return _ptr(a.data)

    def mylab_set_typed_data(self, h, ptr, cls, cplx):
        a = self._get(h)
        if a.cls != cls or a.cplx != bool(cplx):
            return 0
        return self.mylab_set_data(h, ptr, spec.DATA_INTERLEAVED)

    # -- creating arrays -----------------------------------------------------------------------------
    def mxCreateDoubleMatrix(self, m, n, complexity):
        return self._new(DOUBLE, (m, n), complexity)

    def mxCreateDoubleScalar(self, value):
        h = self._new(DOUBLE, (1, 1))
        self.arrays[h].data[0] = value
        return h

    @staticmethod
    def _numeric_class(cls: int) -> int:
        if cls not in DTYPES:
            raise MatlabError(f"Class id {cls} is not a numeric class.", "mylab:mex:badClass")
        return cls

    def mxCreateNumericMatrix(self, m, n, classid, complexity):
        return self._new(self._numeric_class(classid), (m, n), complexity)

    def mxCreateNumericArray(self, ndim, dims, classid, complexity):
        return self._new(self._numeric_class(classid), _sizes(dims, ndim), complexity)

    mxCreateUninitNumericMatrix = mxCreateNumericMatrix
    mxCreateUninitNumericArray = mxCreateNumericArray

    def mxCreateLogicalScalar(self, value):
        h = self._new(LOGICAL, (1, 1))
        self.arrays[h].data[0] = bool(value)
        return h

    def mxCreateLogicalMatrix(self, m, n):
        return self._new(LOGICAL, (m, n))

    def mxCreateLogicalArray(self, ndim, dims):
        return self._new(LOGICAL, _sizes(dims, ndim))

    def mxCreateString(self, text):
        s = _text(text)
        codes = np.frombuffer(s.encode("utf-16-le", "surrogatepass"), dtype=np.uint16).copy()
        a = MxArray(CHAR, (1, codes.size) if codes.size else (0, 0))
        a.data = codes
        return self._register(a)

    def mxCreateCharArray(self, ndim, dims):
        return self._new(CHAR, _sizes(dims, ndim))

    def mxCreateCharMatrixFromStrings(self, m, rows):
        texts = [np.frombuffer(s.encode("utf-16-le", "surrogatepass"), dtype=np.uint16) for s in _strings(rows, m)]
        width = max((t.size for t in texts), default=0)
        grid = np.full((len(texts), width), ord(" "), dtype=np.uint16)
        for i, t in enumerate(texts):
            grid[i, : t.size] = t
        a = MxArray(CHAR, grid.shape)
        a.data = grid.ravel(order="F").copy()
        return self._register(a)

    def mxCreateCellMatrix(self, m, n):
        return self._new(CELL, (m, n))

    def mxCreateCellArray(self, ndim, dims):
        return self._new(CELL, _sizes(dims, ndim))

    def mxCreateStructMatrix(self, m, n, nfields, fieldnames):
        h = self._new(STRUCT, (m, n))
        self._set_struct_fields(h, _strings(fieldnames, nfields))
        return h

    def mxCreateStructArray(self, ndim, dims, nfields, fieldnames):
        h = self._new(STRUCT, _sizes(dims, ndim))
        self._set_struct_fields(h, _strings(fieldnames, nfields))
        return h

    def mxCreateSparse(self, m, n, nzmax, complexity):
        return self._new_sparse(m, n, nzmax, DOUBLE, bool(complexity))

    def mxCreateSparseLogicalMatrix(self, m, n, nzmax):
        return self._new_sparse(m, n, nzmax, LOGICAL, False)

    def mxDuplicateArray(self, h):
        return self._copy(h) if h else None

    def mxDestroyArray(self, h):
        if not h or h not in self.arrays:
            return
        self._detach(h)
        self._destroy(h)

    # -- sizes -------------------------------------------------------------------------------------------
    def mxGetM(self, h):
        return self._get(h).dims[0]

    def mxGetN(self, h):
        return int(np.prod(self._get(h).dims[1:]))

    def mxSetM(self, h, m):
        a = self._get(h)
        a.set_dims((m, *a.dims[1:]))
        self._resize_elements(a, h)

    def mxSetN(self, h, n):
        a = self._get(h)
        a.set_dims((a.dims[0], n))
        if a.sparse and a.jc.size != n + 1:
            jc = np.full(n + 1, a.jc[min(n, a.jc.size - 1)] if a.jc.size else 0, dtype=INDEX)
            k = min(n + 1, a.jc.size)
            jc[:k] = a.jc[:k]
            a.jc = jc
        self._resize_elements(a, h)

    def mxGetNumberOfElements(self, h):
        return self._get(h).numel

    def mxGetNumberOfDimensions(self, h):
        return len(self._get(h).dims)

    def mxGetDimensions(self, h):
        return self._get(h).dims_buf.ctypes.data

    def mxSetDimensions(self, h, dims, ndim):
        a = self._get(h)
        new = _sizes(dims, ndim)
        if a.sparse and len([d for d in new]) > 2:
            return 1
        a.set_dims(new)
        self._resize_elements(a, h)
        return 0

    def mxGetElementSize(self, h):
        a = self._get(h)
        if a.cls in DTYPES:
            return DTYPES[a.cls].itemsize
        if a.cls in (CELL, STRUCT):
            return ctypes.sizeof(ctypes.c_void_p)
        return 0

    def mxCalcSingleSubscript(self, h, nsubs, subs):
        a = self._get(h)
        idx, stride = 0, 1
        for k, s in enumerate(_sizes(subs, nsubs)):
            idx += s * stride
            stride *= a.dims[k] if k < len(a.dims) else 1
        return idx

    def mxIsEmpty(self, h):
        return self._get(h).numel == 0

    def mxIsScalar(self, h):
        return self._get(h).numel == 1

    # -- classes -----------------------------------------------------------------------------------------
    def mxGetClassID(self, h):
        return self._get(h).cls

    def _class_name(self, a: MxArray) -> str:
        return class_name(a.value) if a.value is not None else CLASS_NAMES.get(a.cls, "unknown")

    def mxGetClassName(self, h):
        return self.cstr(self._class_name(self._get(h)))

    def mxIsClass(self, h, name):
        return self._class_name(self._get(h)) == _text(name)

    def mxIsComplex(self, h):
        return self._get(h).cplx

    def mxIsSparse(self, h):
        return self._get(h).sparse

    def mxIsLogicalScalarTrue(self, h):
        a = self._get(h)
        return a.cls == LOGICAL and not a.sparse and a.numel == 1 and bool(a.data[0])

    # -- data ----------------------------------------------------------------------------------------------
    def mxGetPr(self, h):
        return self.mylab_data(h, spec.DATA_REAL)

    def mxGetPi(self, h):
        return self.mylab_data(h, spec.DATA_IMAG)

    def mxSetPr(self, h, ptr):
        self.mylab_set_data(h, ptr, spec.DATA_REAL)

    def mxSetPi(self, h, ptr):
        self.mylab_set_data(h, ptr, spec.DATA_IMAG)

    def mxGetScalar(self, h):
        a = self._get(h)
        if a.data is None or a.data.size == 0 or (a.sparse and int(a.jc[-1]) == 0) or (not a.sparse and a.numel == 0):
            return 0.0
        return float(a.data[0])

    def mxGetIr(self, h):
        a = self._get(h)
        return _ptr(a.ir) if a.sparse else None

    def mxGetJc(self, h):
        a = self._get(h)
        return _ptr(a.jc) if a.sparse else None

    def mxSetIr(self, h, ptr):
        a = self._get(h)
        if a.sparse:
            a.ir = self._view(self._adopt_memory(ptr), INDEX)

    def mxSetJc(self, h, ptr):
        a = self._get(h)
        if a.sparse:
            a.jc = self._view(self._adopt_memory(ptr), INDEX)

    def mxGetNzmax(self, h):
        a = self._get(h)
        return int(a.ir.size) if a.sparse else a.numel

    def mxSetNzmax(self, h, nzmax):
        a = self._get(h)
        if not a.sparse:
            return
        nz = max(int(nzmax), 1)

        def grow(x, n):
            y = np.zeros(n, dtype=x.dtype)
            k = min(n, x.size)
            y[:k] = x[:k]
            return y
        a.ir = grow(a.ir, nz)
        if a.cplx and a.imag is None:
            a.data = grow(a.data, 2 * nz)
        else:
            a.data = grow(a.data, nz)
            if a.imag is not None:
                a.imag = grow(a.imag, nz)

    # -- text ------------------------------------------------------------------------------------------------
    def _char_bytes(self, h) -> bytes | None:
        a = self._get(h)
        if a.cls != CHAR or a.data is None:
            return None
        units = a.data[: a.numel].astype("<u2").tobytes()
        return units.decode("utf-16-le", "surrogatepass").encode("utf-8", "replace")

    def mxArrayToString(self, h):
        raw = self._char_bytes(h)
        if raw is None:
            return None
        addr = self.mxMalloc(len(raw) + 1)
        ctypes.memmove(addr, raw + b"\0", len(raw) + 1)
        return addr

    mxArrayToUTF8String = mxArrayToString

    def mxGetString(self, h, buf, buflen):
        raw = self._char_bytes(h)
        if not buf or buflen <= 0:
            return 1
        if raw is None:
            ctypes.memmove(buf, b"\0", 1)
            return 1
        n = min(len(raw), int(buflen) - 1)
        ctypes.memmove(buf, raw[:n] + b"\0", n + 1)
        return 0 if n == len(raw) else 1

    # -- cells and structs -------------------------------------------------------------------------------------
    def _cell(self, h) -> MxArray:
        a = self._get(h)
        if a.cells is None:
            raise MatlabError("The mxArray is not a cell array.", "mylab:mex:notCell")
        return a

    def _struct(self, h) -> MxArray:
        a = self._get(h)
        if a.fvals is None:
            raise MatlabError("The mxArray is not a struct array.", "mylab:mex:notStruct")
        return a

    @staticmethod
    def _check_index(a: MxArray, i: int, what: str) -> None:
        if not 0 <= i < a.numel:
            raise MatlabError(f"{what}: index {i} is out of range for an array of {a.numel} elements "
                              "(indices start at 0).", "mylab:mex:indexOutOfRange")

    def mxGetCell(self, h, index):
        a = self._get(h)
        if a.cells is None or not 0 <= index < len(a.cells):
            return None
        return a.cells[index] or None

    def mxSetCell(self, h, index, value):
        a = self._cell(h)
        self._check_index(a, index, "mxSetCell")
        old = a.cells[index]
        if old and old != value:
            self._orphan(old)
        a.cells[index] = self._adopt_child(h, value)

    def mxGetField(self, h, index, fieldname):
        a = self._get(h)
        name = _text(fieldname)
        if a.fvals is None or name not in a.fields or not 0 <= index < len(a.fvals):
            return None
        return a.fvals[index][a.fields.index(name)] or None

    def mxSetField(self, h, index, fieldname, value):
        a = self._struct(h)
        name = _text(fieldname)
        k = a.fields.index(name) if name in a.fields else self.mxAddField(h, fieldname)
        if k < 0:
            raise MatlabError(f"mxSetField: '{name}' is not a valid field name.", "MATLAB:InvalidFieldName")
        self.mxSetFieldByNumber(h, index, k, value)

    def mxGetFieldByNumber(self, h, index, fieldnumber):
        a = self._get(h)
        if a.fvals is None or not 0 <= fieldnumber < len(a.fields) or not 0 <= index < len(a.fvals):
            return None
        return a.fvals[index][fieldnumber] or None

    def mxSetFieldByNumber(self, h, index, fieldnumber, value):
        a = self._struct(h)
        self._check_index(a, index, "mxSetFieldByNumber")
        if not 0 <= fieldnumber < len(a.fields):
            raise MatlabError(f"mxSetFieldByNumber: field number {fieldnumber} is out of range (the struct has "
                              f"{len(a.fields)} fields).", "mylab:mex:indexOutOfRange")
        old = a.fvals[index][fieldnumber]
        if old and old != value:
            self._orphan(old)
        a.fvals[index][fieldnumber] = self._adopt_child(h, value)

    def mxGetNumberOfFields(self, h):
        a = self._get(h)
        return len(a.fields) if a.fields is not None else 0

    def mxGetFieldNameByNumber(self, h, fieldnumber):
        a = self._get(h)
        if a.fields is None or not 0 <= fieldnumber < len(a.fields):
            return None
        return self.cstr(a.fields[fieldnumber])

    def mxGetFieldNumber(self, h, fieldname):
        a = self._get(h)
        name = _text(fieldname)
        return a.fields.index(name) if a.fields is not None and name in a.fields else -1

    def mxAddField(self, h, fieldname):
        a = self._get(h)
        name = _text(fieldname)
        if a.fields is None or not _FIELD_NAME.match(name):
            return -1
        if name in a.fields:
            return a.fields.index(name)
        a.fields.append(name)
        for row in a.fvals:
            row.append(0)
        return len(a.fields) - 1

    def mxRemoveField(self, h, fieldnumber):
        a = self._get(h)
        if a.fields is None or not 0 <= fieldnumber < len(a.fields):
            return
        del a.fields[fieldnumber]
        for row in a.fvals:
            c = row.pop(fieldnumber)
            if c:
                self._orphan(c)

    # -- memory ----------------------------------------------------------------------------------------------
    def mxMalloc(self, n):
        buf = np.empty(max(int(n), 1), dtype=np.uint8)
        addr = buf.ctypes.data
        self.mem[addr] = _Block(buf, self._lib())
        if self.scopes:
            self.scopes[-1].mem.add(addr)
        return addr

    def mxCalloc(self, n, size):
        addr = self.mxMalloc(int(n) * int(size))
        self.mem[addr].buf[:] = 0
        return addr

    def mxRealloc(self, ptr, size):
        if not ptr:
            return self.mxMalloc(size)
        old = self.mem.get(ptr)
        if old is None:
            raise MatlabError("mxRealloc: the pointer did not come from mxMalloc, mxCalloc or mxRealloc.",
                              "mylab:mex:foreignPointer")
        addr = self.mxMalloc(size)
        ctypes.memmove(addr, ptr, min(old.buf.size, max(int(size), 1)))
        self.mem[addr].persistent = old.persistent
        del self.mem[ptr]
        return addr

    def mxFree(self, ptr):
        if ptr:
            self.mem.pop(ptr, None)

    # -- calling back into mylab -------------------------------------------------------------------------------
    def _handles(self, ptr, n: int) -> list:
        if n <= 0:
            return []
        if not ptr:
            raise MatlabError("The array of mxArray pointers is NULL.", "mylab:mex:nullPointer")
        return list((ctypes.c_void_p * n).from_address(ptr))

    def _call_mylab(self, nlhs, plhs, nrhs, prhs, name):
        args = [self.from_mx(h) for h in self._handles(prhs, nrhs)]
        fname = _text(name)
        if not fname:
            raise MatlabError("mexCallMATLAB: no function name was given.", "mylab:mex:noFunction")
        interp = self.interp
        outs = interp.call_named(fname, args, nlhs, interp.frame)
        if nlhs > 0:
            if len(outs) < nlhs:
                raise MatlabError(f"Too many output arguments for '{fname}'.", "MATLAB:TooManyOutputs")
            slots = (ctypes.c_void_p * nlhs).from_address(plhs)
            for k in range(nlhs):
                slots[k] = self.to_mx(outs[k])

    def _exception(self, e: MatlabError) -> int:
        from ..interpreter import make_exception
        return self.to_mx(make_exception(e))

    def mexCallMATLAB(self, nlhs, plhs, nrhs, prhs, name):
        try:
            self._call_mylab(nlhs, plhs, nrhs, prhs, name)
        except MatlabError:
            lib = self._lib()
            if lib is not None and lib.trap:
                return 1
            raise
        return 0

    def mexCallMATLABWithTrap(self, nlhs, plhs, nrhs, prhs, name):
        try:
            self._call_mylab(nlhs, plhs, nrhs, prhs, name)
        except MatlabError as e:
            return self._exception(e)
        return None

    def _eval(self, command) -> None:
        from ..parser import parse
        interp = self.interp
        frame = self._workspace(b"caller")
        program = parse(_text(command) + "\n")
        interp.frames.append(frame)
        try:
            interp.run_program(program, frame)
        finally:
            interp.frames.pop()

    def mexEvalString(self, command):
        try:
            self._eval(command)
        except MatlabError:
            lib = self._lib()
            if lib is not None and lib.trap:
                return 1
            raise
        return 0

    def mexEvalStringWithTrap(self, command):
        try:
            self._eval(command)
        except MatlabError as e:
            return self._exception(e)
        return None

    def mexSetTrapFlag(self, flag):
        lib = self._lib()
        if lib is not None:
            lib.trap = bool(flag)

    def mexGetVariable(self, workspace, name):
        frame = self._workspace(workspace)
        var = _text(name)
        if frame == "global":
            if var not in self.interp.global_vars:
                return None
            return self.to_mx(self.interp.global_vars[var])
        if not frame.has(var):
            return None
        return self.to_mx(frame.get(var))

    mexGetVariablePtr = mexGetVariable

    def mexPutVariable(self, workspace, name, h):
        frame = self._workspace(workspace)
        var = _text(name)
        if not _FIELD_NAME.match(var):
            return 1
        value = self.from_mx(h)
        if frame == "global":
            self.interp.global_vars[var] = value
        else:
            frame.set(var, value)
        return 0

    def mexLock(self):
        lib = self._lib()
        if lib is not None:
            lib.locks += 1

    def mexUnlock(self):
        lib = self._lib()
        if lib is not None and lib.locks > 0:
            lib.locks -= 1

    def mexIsLocked(self):
        lib = self._lib()
        return bool(lib is not None and lib.locks)

    def mexFunctionName(self):
        lib = self._lib()
        return self.cstr(lib.name if lib is not None else "")

    def mexMakeArrayPersistent(self, h):
        a = self._get(h)
        a.persistent = True
        a.owner = self._lib()

    def mexMakeMemoryPersistent(self, ptr):
        b = self.mem.get(ptr) if ptr else None
        if b is None:
            raise MatlabError("mexMakeMemoryPersistent: the pointer did not come from mxMalloc, mxCalloc or "
                              "mxRealloc.", "mylab:mex:foreignPointer")
        b.persistent = True
        b.owner = self._lib()


def _too_small() -> MatlabError:
    return MatlabError("The mxArray has less data than its dimensions need (after mxSetM, mxSetN or "
                       "mxSetDimensions, give it a larger buffer with mxSetData).", "mylab:mex:dataTooSmall")


def _as_sparse(v):
    mod = sys.modules.get("mylab.builtins.sparse")
    if mod is not None and isinstance(v, mod.SparseMatrix):
        return v
    return None


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _close(cdll) -> None:
    """Unload a shared library (best effort: the library is no longer used either way)."""
    import _ctypes
    handle = getattr(cdll, "_handle", None)
    if not handle:
        return
    try:
        if os.name == "nt":
            _ctypes.FreeLibrary(handle)
        else:
            _ctypes.dlclose(handle)
    except (OSError, AttributeError):
        pass
