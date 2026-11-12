"""Sparse matrices: storage, operators, indexing, constructors and iterative solvers.

A :class:`SparseMatrix` wraps a SciPy compressed-column matrix (double, complex
or logical). It reports class ``double`` (or ``logical``), as sparse matrices do
in MATLAB; ``issparse`` tells them apart from full arrays.

How functions reach a sparse matrix:

* operators go through :meth:`SparseMatrix.binary_op` / :meth:`unary_op`;
* functions listed in ``_NATIVE`` have a sparse implementation;
* every other built-in receives the full (dense) array, so it still gives the
  right numbers. Conversion of a very large matrix is refused with a clear error
  instead of exhausting memory.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .. import indexing as ix
from .. import ops
from ..display import TIMES, _matrix_lines, format_scalar
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, simplify, to_int, to_matlab, to_numeric,
    to_str,
)
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, is_text, text

DENSE_LIMIT = 2 ** 28  # elements; converting more than this to a full array is refused (2 GiB)

_NATIVE: dict[str, Callable] = {}
_OWN: set[str] = set()  # built-ins defined in this module: they accept sparse inputs themselves

# Built-ins that only inspect or store their arguments: they get the object itself.
_PASSTHROUGH = {
    "size", "numel", "length", "ndims", "isempty", "isscalar", "isvector", "isrow", "iscolumn", "ismatrix",
    "issquare", "disp", "display", "struct", "cell", "deal", "setfield", "getfield", "isfield", "rmfield",
    "fieldnames", "feval", "cellfun", "arrayfun", "structfun", "save", "table", "validateattributes",
    "assert", "error", "warning", "func2str", "builtin", "exist", "isvarname", "whos", "who", "clear",
    "tic", "toc", "timeit", "containers.Map", "isobject", "isvalid", "ishandle", "isgraphics", "nargin",
    "nargout", "inputname", "num2cell_keep", "isdatetime", "isduration", "istable", "iscategorical",
    "isMATLABReleaseOlderThan", "addvars", "array2table",
}
# Type questions: answered from a 1x1 stand-in of the same class, without converting the matrix.
_TYPE_ONLY = {
    "isnumeric", "isfloat", "isreal", "islogical", "isinteger", "ischar", "iscell", "isstruct", "isstring",
    "iscellstr", "is_function_handle", "istext", "isa", "class", "mustBeNumeric", "mustBeNumericOrLogical",
    "mustBeA", "mustBeReal",
}


def native(*names: str):
    def deco(fn: Callable) -> Callable:
        for n in names:
            _NATIVE[n] = fn
        return fn
    return deco


def sbuiltin(*names: str):
    """A built-in defined here that handles sparse and full inputs itself."""
    def deco(fn: Callable) -> Callable:
        _OWN.update(names)
        return builtin(*names)(fn)
    return deco


def _size_text(shape: tuple) -> str:
    return TIMES.join(str(d) for d in shape)


def _canon(m: Any) -> sp.csc_matrix:
    """Canonical storage: CSC, sorted, no stored zeros, dtype bool / float64 / complex128."""
    if not sp.issparse(m):
        m = sp.csc_matrix(np.atleast_2d(np.asarray(m)))
    m = sp.csc_matrix(m)
    kind = m.dtype.kind
    if kind == "c":
        if m.dtype != np.complex128:
            m = m.astype(np.complex128)
        if m.nnz == 0 or not np.any(m.data.imag):
            m = m.real.astype(np.float64)
        else:
            m.data = m.data.real + 1j * (m.data.imag + 0.0)  # no negative zeros in the display
    elif kind != "b" and m.dtype != np.float64:
        m = m.astype(np.float64)
    m.sum_duplicates()
    m.eliminate_zeros()
    if not m.has_sorted_indices:
        m.sort_indices()
    return m


class SparseMatrix(MObject):
    """A 2-D sparse matrix (see the module docstring)."""

    def __init__(self, m: Any, canonical: bool = False):
        self._m = m if canonical else _canon(m)

    # -- storage ------------------------------------------------------------------------------
    @property
    def m(self) -> sp.csc_matrix:
        if self._m.format != "csc":  # left in LIL form by indexed assignment
            self._m = _canon(self._m)
        return self._m

    @property
    def is_logical(self) -> bool:
        return self._m.dtype == bool

    def fm(self) -> sp.csc_matrix:
        """The matrix with a numeric (not logical) dtype."""
        m = self.m
        return m.astype(np.float64) if m.dtype == bool else m

    def to_dense(self) -> np.ndarray:
        r, c = self._m.shape
        if r * c > DENSE_LIMIT:
            gb = r * c * 8 / 2 ** 30
            raise MatlabError(
                f"Out of memory: this operation needs the full form of a {r}{TIMES}{c} sparse matrix "
                f"({gb:.1f} GB). Use a function that works on sparse matrices instead.",
                "MATLAB:array:SizeLimitExceeded")
        return self.m.toarray()

    # duck-typed array attributes (used by for-loops and a few generic helpers)
    @property
    def shape(self) -> tuple:
        return self._m.shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return self._m.shape[0] * self._m.shape[1]

    @property
    def dtype(self):
        return self._m.dtype

    def astype(self, dtype) -> np.ndarray:
        return self.to_dense().astype(dtype)

    def copy(self) -> "SparseMatrix":
        return SparseMatrix(self._m.copy(), canonical=True)

    def __getitem__(self, key):
        return SparseMatrix(self.m[key])

    # -- identity -----------------------------------------------------------------------------
    def class_name(self) -> str:
        return "logical" if self.is_logical else "double"

    def isa_class(self, name: str) -> bool:
        if self.is_logical:
            return name == "logical"
        return name in ("double", "numeric", "float")

    def matlab_shape(self) -> tuple:
        return tuple(int(d) for d in self._m.shape)

    def object_equals(self, other: Any) -> bool:
        return _equal(self, other)

    # -- method dispatch ------------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        if name in _NATIVE:
            return True
        if name in _OWN or name in _PASSTHROUGH:
            return False
        return name in BUILTINS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        fn = _NATIVE.get(name)
        if fn is not None:
            res = fn(interp, args, nargout)
            if res is None:
                return []
            return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]
        user = _user_function(interp, name, args, nargout)
        if user is not None:
            return user
        if name in _TYPE_ONLY:
            stand_in = [np.zeros((1, 1), dtype=a.dtype) if isinstance(a, SparseMatrix) else a for a in args]
            return interp.call_named(name, stand_in, nargout, interp.frame)
        dense = [a.to_dense() if isinstance(a, SparseMatrix) else a for a in args]
        return interp.call_named(name, dense, nargout, interp.frame)

    def get_field(self, interp, name: str) -> Any:
        raise MatlabError("Dot indexing is not supported for variables of type 'double'.")

    # -- operators ------------------------------------------------------------------------------
    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        return _binop(op, left, right)

    def unary_op(self, interp, op: str) -> Any:
        if op == "-":
            return SparseMatrix(-self.fm())
        if op == "+":
            return SparseMatrix(self.fm())
        if op in ("~", "!"):
            return SparseMatrix(sp.csc_matrix(~(self.to_dense() != 0)))
        if op == "'":
            return SparseMatrix(self.m.conj().T)
        if op == ".'":
            return SparseMatrix(self.m.T)
        return NotImplemented

    # -- indexing -------------------------------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        args = _trim_index(args)
        r, c = self._m.shape
        if not args:
            return self
        if len(args) == 1:
            a = args[0]
            if a is ix.COLON:
                return SparseMatrix(_reshape(self.m, (r * c, 1)))
            pos, ishape = _positions(a, r * c)
            if pos.size == 1 and ishape == (1, 1):
                return _dense_scalar(self._m[int(pos[0] % r), int(pos[0] // r)])
            vals = np.asarray(self.m[pos % r, pos // r]).ravel() if pos.size else np.zeros(0, self._m.dtype)
            shape = ix._linear_result_shape((r, c), ishape, pos.size)
            if len(shape) != 2:
                raise MatlabError("N-dimensional indexing of a sparse matrix is not supported.")
            k = np.arange(pos.size)
            rows = k % shape[0] if shape[0] else k
            cols = k // shape[0] if shape[0] else k
            return SparseMatrix(sp.csc_matrix((vals, (rows, cols)), shape=shape))
        if len(args) != 2:
            raise MatlabError("A sparse matrix has two dimensions; use one or two subscripts.")
        rp = None if args[0] is ix.COLON else _positions(args[0], r)[0]
        cp = None if args[1] is ix.COLON else _positions(args[1], c)[0]
        if rp is not None and cp is not None and rp.size == 1 and cp.size == 1:
            return _dense_scalar(self._m[int(rp[0]), int(cp[0])])
        sub = self._m
        if sub.format != "csc" and (rp is None or cp is None or rp.size * cp.size > 4096):
            sub = self.m
        if sub.format == "csc":
            if cp is not None:
                sub = sub[:, cp]
            if rp is not None:
                sub = sub.tocsr()[rp, :]
        else:  # small block of a matrix that is being assembled
            sub = sub[rp.reshape(-1, 1), cp.reshape(1, -1)]
        return SparseMatrix(sub)

    def index_set(self, interp, args: list, value: Any) -> Any:
        target = self if _may_modify_in_place(self, interp) else SparseMatrix(self._m.copy(), canonical=True)
        args = _trim_index(args)
        if isinstance(value, SparseMatrix):
            val = value.fm() if not (self.is_logical and value.is_logical) else value.m
        else:
            if isinstance(value, (Cell, Struct, FunctionHandle, MString, StringArray)) or isinstance(value, MObject):
                raise MatlabError("Only numeric or logical values can be stored in a sparse matrix.")
            val = to_numeric(value)
            if val.ndim > 2:
                raise MatlabError("Sparse matrices are two-dimensional.")
            if val.dtype.kind in "iu" or val.dtype == np.float32:
                val = val.astype(float)
        m = target._m
        if m.dtype == bool and val.dtype != bool:
            m = m.astype(np.float64)
        if val.dtype.kind == "c" and m.dtype.kind != "c":
            m = m.astype(np.complex128)
        if m.format != "lil":
            m = m.tolil()
        r, c = m.shape
        scalar_val = val.shape == (1, 1)
        if scalar_val:
            val = val[0, 0] if not sp.issparse(val) else val.toarray()[0, 0]
        if len(args) == 1:
            a = args[0]
            if a is ix.COLON:
                pos, count = np.arange(r * c), r * c
            else:
                pos, _ = _positions(a, r * c, for_set=True)
                count = pos.size
            if pos.size and pos.max() >= r * c:
                if r > 1 and c > 1:
                    raise MatlabError("Attempt to grow a matrix along an ambiguous dimension with a linear "
                                      "index. Use two subscripts.")
                new = (int(pos.max()) + 1, 1) if (c == 1 and r != 1) else (1, int(pos.max()) + 1)
                if r * c == 0:
                    new = (1, int(pos.max()) + 1)
                m.resize(new)
                r, c = new
            if not scalar_val:
                vals = (val.toarray() if sp.issparse(val) else np.asarray(val)).flatten(order="F")
                if vals.size != count:
                    raise MatlabError("Unable to perform assignment because the left and right sides have a "
                                      "different number of elements.")
            else:
                vals = val
            if count:
                m[pos % r, pos // r] = vals
        elif len(args) == 2:
            rp = np.arange(r) if args[0] is ix.COLON else _positions(args[0], r, for_set=True)[0]
            cp = np.arange(c) if args[1] is ix.COLON else _positions(args[1], c, for_set=True)[0]
            if not scalar_val:
                if args[0] is ix.COLON and r == 0 and c == 0:
                    rp = np.arange(val.shape[0])
                if args[1] is ix.COLON and r == 0 and c == 0:
                    cp = np.arange(val.shape[1])
            nr = max(r, int(rp.max()) + 1 if rp.size else 0)
            nc = max(c, int(cp.max()) + 1 if cp.size else 0)
            if (nr, nc) != (r, c):
                m.resize((nr, nc))
            if scalar_val:
                if rp.size == 1 and cp.size == 1:
                    m[int(rp[0]), int(cp[0])] = val
                elif rp.size and cp.size:
                    m[rp.reshape(-1, 1), cp.reshape(1, -1)] = val
            else:
                if val.shape != (rp.size, cp.size):
                    if val.shape[0] * val.shape[1] == rp.size * cp.size and 1 in val.shape \
                            and 1 in (rp.size, cp.size):
                        val = val.reshape((rp.size, cp.size))
                    else:
                        raise MatlabError(
                            f"Unable to perform assignment because the size of the left side is "
                            f"{rp.size}-by-{cp.size} and the size of the right side is "
                            f"{val.shape[0]}-by-{val.shape[1]}.", "MATLAB:subsassigndimmismatch")
                if rp.size and cp.size:
                    m[rp.reshape(-1, 1), cp.reshape(1, -1)] = val
        else:
            raise MatlabError("A sparse matrix has two dimensions; use one or two subscripts.")
        target._m = m
        return target

    def index_delete(self, interp, args: list) -> Any:
        args = _trim_index(args)
        m = self.m
        r, c = m.shape
        if len(args) == 1:
            if args[0] is ix.COLON:
                return SparseMatrix(sp.csc_matrix((0, 0)))
            pos, _ = _positions(args[0], r * c)
            keep = np.ones(r * c, dtype=bool)
            keep[pos] = False
            flat = _reshape(m, (r * c, 1))[np.nonzero(keep)[0], :]
            return SparseMatrix(flat if (c == 1 and r != 1) else flat.T)
        if len(args) != 2:
            raise MatlabError("A sparse matrix has two dimensions; use one or two subscripts.")
        full_rows = args[0] is ix.COLON or ix._covers(args[0], r)
        full_cols = args[1] is ix.COLON or ix._covers(args[1], c)
        if full_rows and not full_cols:
            keep = np.ones(c, dtype=bool)
            keep[_positions(args[1], c)[0]] = False
            return SparseMatrix(m[:, np.nonzero(keep)[0]])
        if full_cols:
            if args[0] is ix.COLON:
                return SparseMatrix(sp.csc_matrix((0, c)))
            keep = np.ones(r, dtype=bool)
            keep[_positions(args[0], r)[0]] = False
            return SparseMatrix(m.tocsr()[np.nonzero(keep)[0], :])
        raise MatlabError("A null assignment can have only one non-colon index.")

    # -- concatenation ----------------------------------------------------------------------------
    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        mats = []
        for it in items:
            if isinstance(it, SparseMatrix):
                mats.append(it.m)
                continue
            if not isinstance(it, np.ndarray):
                raise MatlabError("Only numeric or logical arrays can be concatenated with a sparse matrix.")
            if it.shape == (0, 0):
                continue
            a = to_numeric(it)
            if a.ndim > 2:
                raise MatlabError("Sparse matrices are two-dimensional.")
            mats.append(sp.csc_matrix(a if it.dtype == bool or a.dtype.kind == "c" else a.astype(float))
                        if it.dtype != bool else sp.csc_matrix(it))
        mats = [x for x in mats if x.shape != (0, 0)]
        if not mats:
            return SparseMatrix(sp.csc_matrix((0, 0)))
        other = 1 - axis
        if len({x.shape[other] for x in mats}) > 1:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch")
        if not all(x.dtype == bool for x in mats):
            mats = [x.astype(np.float64) if x.dtype == bool else x for x in mats]
        out = sp.hstack(mats, format="csc") if axis == 1 else sp.vstack(mats, format="csc")
        return SparseMatrix(out)

    # -- display --------------------------------------------------------------------------------
    def _kind(self) -> str:
        if self.is_logical:
            return "logical"
        return "complex double" if self._m.dtype.kind == "c" else "double"

    def summary_text(self, in_cell: bool) -> str:
        s = f"{_size_text(self.matlab_shape())} sparse {self._kind()}"
        return s if in_cell else f"[{s}]"

    def display_lines(self) -> list[str]:
        m = self.m
        if m.nnz == 0:
            return [f"   All zero sparse: {_size_text(m.shape)}"]
        coo = m.tocoo()
        order = np.lexsort((coo.row, coo.col))
        limit = 20000
        rows, cols, vals = coo.row[order][:limit], coo.col[order][:limit], coo.data[order][:limit]
        labels = [f"({i + 1},{j + 1})" for i, j in zip(rows, cols)]
        width = max(len(s) for s in labels)
        if vals.dtype == bool:
            texts = ["   1"] * len(labels)
        else:
            texts = _matrix_lines(vals.reshape(-1, 1))
            if len(texts) != len(labels):  # a common scale factor was pulled out: format one by one
                cells = [format_scalar(v) for v in vals]
                w = max(len(s) for s in cells)
                texts = [" " * 3 + s.rjust(w) for s in cells]
        lines = [f"   {lab:<{width}}  {t}" for lab, t in zip(labels, texts)]
        if m.nnz > limit:
            lines.append(f"   ... ({m.nnz - limit} more nonzeros not shown)")
        return lines

    def format_display(self, name: str) -> str:
        m = self.m
        if m.nnz == 0:
            return f"{name} =\n\n" + "\n".join(self.display_lines()) + "\n\n"
        head = (f"  {_size_text(m.shape)} sparse {self._kind()} matrix "
                f"({m.nnz} nonzero{'s' if m.nnz != 1 else ''})")
        return f"{name} =\n\n{head}\n\n" + "\n".join(self.display_lines()) + "\n\n"


# -- small helpers --------------------------------------------------------------------------------

def _reshape(m: sp.csc_matrix, shape: tuple) -> sp.csc_matrix:
    """Column-major reshape that works for index ranges beyond 32 bits."""
    coo = m.tocoo()
    lin = coo.row.astype(np.int64) + coo.col.astype(np.int64) * m.shape[0]
    rows = lin % shape[0] if shape[0] else lin
    cols = lin // shape[0] if shape[0] else lin
    return sp.csc_matrix((coo.data, (rows, cols)), shape=shape)


def _dense_scalar(v: Any) -> np.ndarray:
    v = np.asarray(v).reshape(-1)[0] if not np.isscalar(v) else v
    if isinstance(v, (bool, np.bool_)):
        return np.array([[bool(v)]])
    if isinstance(v, (complex, np.complexfloating)) and v.imag != 0:
        return np.array([[complex(v)]])
    return np.array([[float(np.real(v))]])


def _trim_index(args: list) -> list:
    """Drop trailing subscripts equal to 1 (S(i,j,1) is S(i,j))."""
    args = list(args)
    while len(args) > 2:
        last = args[-1]
        if isinstance(last, np.ndarray) and last.size == 1 and last.dtype != bool and float(np.real(last.flat[0])) == 1:
            args.pop()
        else:
            raise MatlabError("Index exceeds the number of dimensions of a sparse matrix (2).")
    return args


def _positions(idx: Any, extent: int, for_set: bool = False):
    if isinstance(idx, SparseMatrix):
        if idx.is_logical:
            coo = idx.m.tocoo()
            pos = np.sort(coo.row.astype(np.int64) + coo.col.astype(np.int64) * idx.shape[0])
            if pos.size and pos[-1] >= extent and not for_set:
                raise MatlabError("The logical indices contain a true value outside of the array bounds.")
            return pos, ((1, pos.size) if idx.shape[0] == 1 else (pos.size, 1))
        idx = idx.to_dense()
    return ix._to_positions(idx, extent, for_set=for_set)


_REF = {"baseline": None, "probe": None}


def _calibrate() -> None:
    """Find the reference count a matrix has when one plain variable holds it during ``S(i,j) = v``."""
    from ..interpreter import Interpreter
    _REF["probe"] = []
    try:
        it = Interpreter(out=lambda s: None)
        it.execute("K__ = sparse(2, 2); K__(1, 1) = 1;")
        _REF["baseline"] = _REF["probe"][0] if _REF["probe"] else -1
    except Exception:  # pragma: no cover - never fatal: just always copy
        _REF["baseline"] = -1
    finally:
        _REF["probe"] = None


def _may_modify_in_place(obj: "SparseMatrix", interp) -> bool:
    """True when ``obj`` is held by exactly one variable, so assembly loops need not copy it.

    Value semantics must hold (``B = A; B(1,1) = 5`` leaves A alone), so this is
    conservative: the object must be bound directly to one variable of the running
    workspace and have exactly the reference count of that simple case.
    """
    rc = sys.getrefcount(obj)
    if _REF["probe"] is not None:
        _REF["probe"].append(rc)
        return False
    if _REF["baseline"] is None:
        _calibrate()
    if rc != _REF["baseline"]:
        return False
    try:
        frame_vars = interp.frame.vars
    except Exception:  # pragma: no cover
        return False
    return sum(1 for v in frame_vars.values() if v is obj) == 1


def _user_function(interp, name: str, args: list, nargout: int):
    """Call a user-defined function that shadows a built-in name, keeping sparse arguments sparse."""
    try:
        frame = interp.frame
        fn = frame.functions.get(name)
        if fn is not None:
            if hasattr(fn, "fn") and hasattr(fn, "frame"):  # nested function reference
                return interp.call_user_function(fn.fn, args, nargout, frame.functions, frame.file, parent=fn.frame)
            return interp.call_user_function(fn, args, nargout, frame.functions, frame.file)
        found = interp.find_file_function(name)
    except AttributeError:  # pragma: no cover
        return None
    if found is not None:
        program, path = found
        if program.functions and not program.body and not program.classdefs:
            main = next(iter(program.functions.values()))
            return interp.call_user_function(main, args, nargout, program.functions, path)
    return None


def is_sparse(v: Any) -> bool:
    return isinstance(v, SparseMatrix)


def _dense(v: Any) -> np.ndarray:
    if isinstance(v, SparseMatrix):
        return v.to_dense()
    a = to_numeric(v)
    return a.astype(float) if a.dtype.kind in "iu" or a.dtype == np.float32 else a


def as_csc(v: Any) -> sp.csc_matrix:
    """Any numeric value as a canonical SciPy matrix (logical stays logical)."""
    if isinstance(v, SparseMatrix):
        return v.m
    if isinstance(v, np.ndarray) and v.dtype == bool:
        return _canon(sp.csc_matrix(v))
    a = to_numeric(v)
    if a.ndim > 2:
        raise MatlabError("Sparse matrices are two-dimensional.")
    return _canon(sp.csc_matrix(a))


def _num_csc(v: Any) -> sp.csc_matrix:
    m = as_csc(v)
    return m.astype(np.float64) if m.dtype == bool else m


def _shape(v: Any) -> tuple:
    return v.shape if isinstance(v, (SparseMatrix, np.ndarray)) else np.shape(to_numeric(v))


def _is_scalar(v: Any) -> bool:
    return _shape(v) == (1, 1)


def _scalar_value(v: Any):
    if isinstance(v, SparseMatrix):
        x = v.fm()[0, 0]
    else:
        x = to_numeric(v)[0, 0]
    x = complex(x)
    return x.real if x.imag == 0 else x


def _maybe_scalar(m: Any) -> Any:
    """Reductions return a plain number when the result is 1x1."""
    m = _canon(m)
    if m.shape == (1, 1):
        return _dense_scalar(m[0, 0])
    return SparseMatrix(m, canonical=True)


def _warn(interp, msg: str, identifier: str = "") -> None:
    if interp is not None:
        interp.warn(msg, identifier)
    else:
        ops._warn(msg, identifier)


# -- binary operators ------------------------------------------------------------------------------

def _dense_op(op: str, a: Any, b: Any) -> np.ndarray:
    return ops.binop(op, _dense(a), _dense(b))


def _sparse_from_dense_op(op: str, a: Any, b: Any) -> "SparseMatrix":
    return SparseMatrix(sp.csc_matrix(_dense_op(op, a, b)))


def _check_numeric(v: Any) -> None:
    if not isinstance(v, (SparseMatrix, np.ndarray)):
        to_numeric(v)  # raises the usual error for cells, structs, strings
        raise MatlabError("Operands to sparse matrix operations must be numeric or logical.")
    if isinstance(v, np.ndarray) and v.ndim > 2:
        raise MatlabError("Sparse matrices cannot be combined with N-dimensional arrays.")


def _with_data(m: sp.csc_matrix, data: np.ndarray) -> "SparseMatrix":
    out = sp.csc_matrix((data, m.indices.copy(), m.indptr.copy()), shape=m.shape)
    return SparseMatrix(out)


def _times(a: Any, b: Any) -> "SparseMatrix":
    sa, sb = _is_scalar(a), _is_scalar(b)
    if sa or sb:
        k, other = (_scalar_value(a), b) if sa else (_scalar_value(b), a)
        if np.isfinite(k) and isinstance(other, SparseMatrix):
            m = other.fm()
            return _with_data(m, m.data * k)
        return _sparse_from_dense_op(".*", a, b)
    A, B = _shape(a), _shape(b)
    if isinstance(a, SparseMatrix) and isinstance(b, SparseMatrix):
        if A == B:
            return SparseMatrix(a.fm().multiply(b.fm()))
        return _sparse_from_dense_op(".*", a, b)
    s, d = (a, b) if isinstance(a, SparseMatrix) else (b, a)
    dd = _dense(d)
    ops._check_broadcast(np.empty(A), np.empty(B), ".*")
    if np.all(np.isfinite(dd)) and (dd.shape[0] <= s.shape[0] and dd.shape[1] <= s.shape[1]):
        return SparseMatrix(s.fm().multiply(dd))
    return _sparse_from_dense_op(".*", a, b)


def _rdivide(a: Any, b: Any) -> "SparseMatrix":
    if isinstance(a, SparseMatrix) and not isinstance(b, SparseMatrix):
        dd = _dense(b)
        if np.all(dd != 0) and not np.any(np.isnan(dd)):
            m = a.fm()
            if dd.shape == (1, 1):
                with np.errstate(all="ignore"):
                    return _with_data(m, m.data / dd[0, 0])
            ops._check_broadcast(np.empty(a.shape), np.empty(dd.shape), "./")
            if dd.shape[0] <= a.shape[0] and dd.shape[1] <= a.shape[1]:
                with np.errstate(all="ignore"):
                    return SparseMatrix(m.multiply(1.0 / dd))
    return _sparse_from_dense_op("./", a, b)


def _power(a: Any, b: Any) -> "SparseMatrix":
    if isinstance(a, SparseMatrix) and not isinstance(b, SparseMatrix) and _is_scalar(b):
        p = _scalar_value(b)
        m = a.fm()
        if not isinstance(p, complex) and p > 0 and np.isfinite(p) and \
                (m.dtype.kind != "c" and (p == int(p) or m.nnz == 0 or m.data.min() >= 0)):
            with np.errstate(all="ignore"):
                return _with_data(m, m.data ** p)
    return _sparse_from_dense_op(".^", a, b)


def _mtimes(a: Any, b: Any) -> Any:
    if _is_scalar(a) or _is_scalar(b):
        return _times(a, b)
    A, B = _shape(a), _shape(b)
    if A[1] != B[0]:
        raise MatlabError(
            "Incorrect dimensions for matrix multiplication. Check that the number of columns in the "
            "first matrix matches the number of rows in the second matrix. To operate on each element "
            "of the matrix individually, use TIMES (.*) for elementwise multiplication.", "MATLAB:innerdim")
    if isinstance(a, SparseMatrix) and isinstance(b, SparseMatrix):
        return SparseMatrix(a.fm() @ b.fm())
    if isinstance(a, SparseMatrix):
        return simplify(np.asarray(a.fm() @ _dense(b)))
    return simplify(np.asarray((b.fm().T @ _dense(a).T).T))


def _solve(interp, a: Any, b: Any) -> Any:
    """a \\ b where at least one operand is sparse."""
    if _is_scalar(a):
        return _rdivide(b, a) if isinstance(b, SparseMatrix) else _dense_op("./", b, a)
    A, B = _shape(a), _shape(b)
    if A[0] != B[0]:
        raise MatlabError("Matrix dimensions must agree: the number of rows of the two operands of \\ "
                          "must be the same.", "MATLAB:dimagree")
    if not isinstance(a, SparseMatrix):
        return ops.left_divide(_dense(a), _dense(b))
    m = a.fm()
    b_sparse = isinstance(b, SparseMatrix)
    if A[0] != A[1]:  # least squares
        if A[0] * A[1] <= 4_000_000:
            x = ops.left_divide(m.toarray(), _dense(b))
        else:
            rhs = _dense(b)
            cols = [spla.lsqr(m, rhs[:, k], atol=1e-13, btol=1e-13, iter_lim=20 * max(A))[0]
                    for k in range(rhs.shape[1])]
            x = np.column_stack(cols) if cols else np.zeros((A[1], 0))
        return SparseMatrix(sp.csc_matrix(x)) if b_sparse else x
    if A[0] == 0:
        return np.zeros((0, B[1]))
    rhs = b.fm() if b_sparse else _dense(b)
    if m.dtype.kind != "c" and rhs.dtype.kind == "c":
        m = m.astype(np.complex128)
    elif m.dtype.kind == "c" and rhs.dtype.kind != "c":
        rhs = rhs.astype(np.complex128)
    singular = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", spla.MatrixRankWarning)
        warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)
        try:
            if b_sparse:
                x = spla.spsolve(m, rhs)
            else:
                x = spla.splu(m).solve(np.ascontiguousarray(rhs))
        except (spla.MatrixRankWarning, RuntimeError):
            singular = True
    if singular:
        _warn(interp, "Matrix is singular to working precision.", "MATLAB:singularMatrix")
        with np.errstate(all="ignore"):
            x = np.linalg.lstsq(m.toarray(), _dense(b), rcond=None)[0] if A[0] * A[1] <= 4_000_000 \
                else np.full(B, np.inf)
        return SparseMatrix(sp.csc_matrix(x)) if b_sparse else x
    if b_sparse:
        if not sp.issparse(x):
            x = sp.csc_matrix(np.asarray(x).reshape(B, order="F"))
        return SparseMatrix(x)
    return simplify(np.asarray(x).reshape(B, order="F"))


_ZERO_VS = {"==": lambda k: k == 0, "~=": lambda k: k != 0, "<": lambda k: 0 < k, "<=": lambda k: 0 <= k,
            ">": lambda k: 0 > k, ">=": lambda k: 0 >= k}
_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "~=": "~="}


def _compare(op: str, a: Any, b: Any) -> "SparseMatrix":
    if isinstance(b, SparseMatrix) and not isinstance(a, SparseMatrix):
        return _compare(_FLIP[op], b, a)
    # a is sparse
    if not isinstance(b, SparseMatrix) and _is_scalar(b):
        k = to_numeric(b)[0, 0]
        kr = float(np.real(k))
        zero_true = bool(_ZERO_VS[op](k if op in ("==", "~=") else kr))
        if not zero_true and not np.isnan(kr):
            m = a.fm()
            mask = ops.compare(op, m.data.reshape(-1, 1), np.array([[k]])).ravel()
            return _with_data(m, mask.astype(bool))
    elif isinstance(b, SparseMatrix) and a.shape == b.shape and op in ("<", ">", "~="):
        ma, mb = a.fm(), b.fm()
        if ma.dtype.kind != "c" and mb.dtype.kind != "c":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = {"<": lambda: ma < mb, ">": lambda: ma > mb, "~=": lambda: ma != mb}[op]()
            return SparseMatrix(res)
    return SparseMatrix(sp.csc_matrix(ops.compare(op, _dense(a), _dense(b))))


def _logic(op: str, a: Any, b: Any) -> Any:
    if isinstance(a, SparseMatrix) and isinstance(b, SparseMatrix) and a.shape == b.shape:
        x, y = (a.m != 0), (b.m != 0)
        if op == "&":
            return SparseMatrix(x.multiply(y).astype(bool))
        return SparseMatrix((x.astype(np.int8) + y.astype(np.int8)).astype(bool))
    da = _dense(a) != 0
    db = _dense(b) != 0
    res = ops.binop(op, da, db)
    return SparseMatrix(sp.csc_matrix(res)) if op == "&" else res


def _binop(op: str, a: Any, b: Any, interp=None) -> Any:
    _check_numeric(a)
    _check_numeric(b)
    both = isinstance(a, SparseMatrix) and isinstance(b, SparseMatrix)
    if op in ("+", "-"):
        if both and a.shape == b.shape:
            return SparseMatrix(a.fm() + b.fm() if op == "+" else a.fm() - b.fm())
        if both:
            return _sparse_from_dense_op(op, a, b)
        return _dense_op(op, a, b)
    if op == ".*":
        return _times(a, b)
    if op == "*":
        return _mtimes(a, b)
    if op == "./":
        return _rdivide(a, b)
    if op == ".\\":
        return _rdivide(b, a)
    if op == ".^":
        return _power(a, b)
    if op == "\\":
        return _solve(interp, a, b)
    if op == "/":
        if _is_scalar(b):
            return _rdivide(a, b)
        x = _solve(interp, _transpose(b), _transpose(a))
        return _transpose(x)
    if op == "^":
        if isinstance(a, SparseMatrix) and not isinstance(b, SparseMatrix) and _is_scalar(b):
            p = _scalar_value(b)
            if a.shape[0] != a.shape[1]:
                raise MatlabError("Incorrect dimensions for raising a matrix to a power. Check that the matrix "
                                  "is square. To raise each element to a power, use POWER (.^).", "MATLAB:square")
            if not isinstance(p, complex) and p == int(p) and p >= 0:
                return SparseMatrix(spla.matrix_power(a.fm(), int(p)))
        return ops.binop("^", _dense(a), _dense(b))
    if op in _ZERO_VS:
        return _compare(op, a, b)
    if op in ("&", "|"):
        return _logic(op, a, b)
    return NotImplemented


def _transpose(v: Any) -> Any:
    if isinstance(v, SparseMatrix):
        return SparseMatrix(v.m.T)
    return np.asarray(v).T


def _equal(a: Any, b: Any) -> bool:
    for v in (a, b):
        if not isinstance(v, SparseMatrix) and not (isinstance(v, np.ndarray) and v.dtype.kind in "biufc"):
            return False
    if _shape(a) != _shape(b):
        return False
    ma, mb = _num_csc(a), _num_csc(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (ma != mb).nnz == 0


# -- constructors ------------------------------------------------------------------------------------

def _index_vector(v: Any, what: str) -> np.ndarray:
    a = _dense(v).flatten(order="F")
    a = np.real(a)
    if a.size and (np.any(a < 1) or np.any(a != np.floor(a)) or not np.all(np.isfinite(a))):
        raise MatlabError(f"Sparse matrix {what} indices must be positive integers.", "MATLAB:sparse:indexOutOfRange")
    return a.astype(np.int64) - 1


@sbuiltin("sparse")
def _sparse(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for sparse.")
    if len(args) == 1:
        v = args[0]
        if isinstance(v, SparseMatrix):
            return v
        if not isinstance(v, np.ndarray):
            raise MatlabError("sparse(A) needs a numeric or logical matrix.")
        if v.ndim > 2:
            raise MatlabError("N-dimensional arrays cannot be converted to sparse.")
        return SparseMatrix(as_csc(v))
    if len(args) == 2:
        r, c = to_int(args[0], "row count"), to_int(args[1], "column count")
        if r < 0 or c < 0:
            raise MatlabError("Sparse matrix sizes must be non-negative integers.")
        return SparseMatrix(sp.csc_matrix((r, c)))
    if len(args) not in (3, 5, 6):
        raise MatlabError("sparse expects 1, 2, 3, 5 or 6 inputs: sparse(A), sparse(m,n), sparse(i,j,v) or "
                          "sparse(i,j,v,m,n).")
    i = _index_vector(args[0], "row")
    j = _index_vector(args[1], "column")
    vraw = args[2].to_dense() if isinstance(args[2], SparseMatrix) else args[2]
    logical_v = isinstance(vraw, np.ndarray) and vraw.dtype == bool
    v = (vraw if logical_v else _dense(vraw)).flatten(order="F")
    n = max(i.size, j.size, v.size)
    for x in (i, j, v):
        if x.size not in (1, n):
            raise MatlabError("Vectors must be the same length (sparse(i,j,v): i, j and v need matching sizes).",
                              "MATLAB:samelen")
    i, j, v = (np.broadcast_to(x, (n,)) if x.size == 1 else x for x in (i, j, v))
    if len(args) >= 5:
        r, c = to_int(args[3], "row count"), to_int(args[4], "column count")
        if n and (i.max() >= r or j.max() >= c):
            raise MatlabError("Index exceeds matrix dimensions (sparse(i,j,v,m,n)).", "MATLAB:sparse:indexOutOfRange")
    else:
        r = int(i.max()) + 1 if n else 0
        c = int(j.max()) + 1 if n else 0
    if logical_v:
        m = sp.csc_matrix((v.astype(np.int8), (i, j)), shape=(r, c)).astype(bool)
    else:
        m = sp.csc_matrix((v, (i, j)), shape=(r, c))
    return SparseMatrix(m)


@sbuiltin("full")
def _full(interp, args, nargout):
    v = args[0]
    return v.to_dense() if isinstance(v, SparseMatrix) else v


@sbuiltin("issparse")
def _issparse(interp, args, nargout):
    return bool_scalar(bool(args) and isinstance(args[0], SparseMatrix))


def _dims(args: list, name: str) -> tuple[int, int]:
    if not args:
        return 1, 1
    if len(args) == 1:
        a = _dense(args[0]).flatten()
        if a.size == 1:
            return int(a[0]), int(a[0])
        if a.size == 2:
            return int(a[0]), int(a[1])
        raise MatlabError(f"{name}: the size must be a scalar or a two-element vector.")
    return to_int(args[0], "row count"), to_int(args[1], "column count")


@sbuiltin("speye")
def _speye(interp, args, nargout):
    r, c = _dims(args, "speye")
    return SparseMatrix(sp.eye(max(r, 0), max(c, 0), format="csc"))


@sbuiltin("spalloc")
def _spalloc(interp, args, nargout):
    r, c = _dims(args[:2], "spalloc")
    return SparseMatrix(sp.csc_matrix((r, c)))


@sbuiltin("spones")
def _spones(interp, args, nargout):
    m = _num_csc(args[0])
    return _with_data(m, np.ones(m.nnz))


@sbuiltin("spfun")
def _spfun(interp, args, nargout):
    m = _num_csc(args[1])
    if m.nnz == 0:
        return SparseMatrix(m)
    res = interp.call_function_value(args[0], [m.data.reshape(-1, 1).copy()], 1)
    vals = to_numeric(res[0]).flatten(order="F")
    if vals.size != m.nnz:
        raise MatlabError("spfun: the function must return one value per nonzero element.")
    return _with_data(m, vals)


@sbuiltin("nzmax")
def _nzmax(interp, args, nargout):
    v = args[0]
    if isinstance(v, SparseMatrix):
        return float(max(v.m.nnz, 1))
    return float(np.prod(np.shape(to_numeric(v))))


@sbuiltin("nonzeros")
def _nonzeros(interp, args, nargout):
    v = args[0]
    if isinstance(v, SparseMatrix):
        return v.m.data.reshape(-1, 1).copy()
    a = v if isinstance(v, np.ndarray) and v.dtype == bool else to_numeric(v)
    flat = a.flatten(order="F")
    return flat[flat != 0].reshape(-1, 1)


@sbuiltin("spconvert")
def _spconvert(interp, args, nargout):
    d = _dense(args[0])
    if d.ndim != 2 or d.shape[1] not in (3, 4):
        raise MatlabError("spconvert expects a matrix with three or four columns: [i j value].")
    vals = d[:, 2] if d.shape[1] == 3 else d[:, 2] + 1j * d[:, 3]
    i = _index_vector(d[:, 0], "row")
    j = _index_vector(d[:, 1], "column")
    return SparseMatrix(sp.csc_matrix((vals, (i, j))))


def _rand_pattern(interp, args: list, name: str, sampler) -> "SparseMatrix":
    from .core import _rng
    rng = _rng(interp)
    if len(args) == 1:
        m = _num_csc(args[0])
        return _with_data(m, sampler(rng, m.nnz))
    if len(args) < 3:
        raise MatlabError(f"{name} expects {name}(S) or {name}(m, n, density).")
    r, c = to_int(args[0], "row count"), to_int(args[1], "column count")
    density = float(np.real(to_numeric(args[2]).flat[0]))
    if not 0 <= density <= 1:
        raise MatlabError(f"{name}: the density must be between 0 and 1.")
    total = r * c
    k = int(round(density * total))
    if k == 0 or total == 0:
        return SparseMatrix(sp.csc_matrix((r, c)))
    if total <= 50_000_000:
        lin = rng.choice(total, size=k, replace=False)
    else:
        lin = np.unique(rng.integers(0, total, size=k))
    vals = sampler(rng, lin.size)
    return SparseMatrix(sp.csc_matrix((vals, (lin % r, lin // r)), shape=(r, c)))


@sbuiltin("sprand")
def _sprand(interp, args, nargout):
    return _rand_pattern(interp, args, "sprand", lambda rng, k: rng.random(k))


@sbuiltin("sprandn")
def _sprandn(interp, args, nargout):
    return _rand_pattern(interp, args, "sprandn", lambda rng, k: rng.standard_normal(k))


@sbuiltin("sprandsym")
def _sprandsym(interp, args, nargout):
    if len(args) == 1:
        m = _num_csc(args[0])
        low = sp.tril(_rand_pattern(interp, [SparseMatrix(m)], "sprandsym",
                                    lambda rng, k: rng.standard_normal(k)).m)
    else:
        low = sp.tril(_rand_pattern(interp, [args[0], args[0], args[1]], "sprandsym",
                                    lambda rng, k: rng.standard_normal(k)).m)
    return SparseMatrix(low + sp.tril(low, -1).T)


@sbuiltin("spdiags")
def _spdiags(interp, args, nargout):
    if len(args) <= 2:  # extract
        m = _num_csc(args[0])
        r, c = m.shape
        coo = m.tocoo()
        if len(args) == 2:
            d = np.real(_dense(args[1])).flatten().astype(int)
        else:
            d = np.unique(coo.col.astype(np.int64) - coo.row)
        nrow = min(r, c)
        out = np.zeros((nrow, d.size), dtype=m.dtype)
        for k, dk in enumerate(d):
            diag = m.diagonal(int(dk))
            if diag.size == 0:
                continue
            # position of the diagonal's first element inside the output column
            start = (max(dk, 0) if r >= c else max(-dk, 0))
            out[start:start + diag.size, k] = diag
        if len(args) == 2:
            return out
        return (out, d.reshape(-1, 1).astype(float)) if nargout > 1 else out
    b = _dense(args[0])
    d = np.real(_dense(args[1])).flatten().astype(int)
    if len(args) == 3:
        base = _num_csc(args[2]).tolil()
        r, c = base.shape
    elif len(args) == 4:
        r, c = to_int(args[2], "row count"), to_int(args[3], "column count")
        base = None
    else:
        raise MatlabError("spdiags expects spdiags(A), spdiags(A, d), spdiags(B, d, A) or spdiags(B, d, m, n).")
    if b.shape[0] == 1 and b.shape[1] != d.size and b.shape[1] > 1 and d.size == 1:
        b = b.reshape(-1, 1)
    if b.shape[1] != d.size:
        if b.size == d.size and d.size > 1 and b.shape[0] != 1:
            b = b.reshape(1, -1)
        else:
            raise MatlabError("spdiags: B needs one column per diagonal listed in d.")
    rows_l, cols_l, vals_l = [], [], []
    for k, dk in enumerate(d):
        i = np.arange(max(0, -dk), min(r, c - dk))
        if i.size == 0:
            continue
        j = i + dk
        src = j if r >= c else i
        if b.shape[0] == 1:
            vals = np.full(i.size, b[0, k])
        else:
            if src.max() >= b.shape[0]:
                raise MatlabError("spdiags: B has too few rows for the requested diagonals.")
            vals = b[src, k]
        rows_l.append(i)
        cols_l.append(j)
        vals_l.append(vals)
    if base is not None:
        for i, j, vals in zip(rows_l, cols_l, vals_l):
            base[i, j] = vals
        return SparseMatrix(base)
    if not rows_l:
        return SparseMatrix(sp.csc_matrix((r, c)))
    return SparseMatrix(sp.csc_matrix((np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
                                      shape=(r, c)))


@sbuiltin("blkdiag")
def _blkdiag(interp, args, nargout):
    if any(isinstance(a, SparseMatrix) for a in args):
        return SparseMatrix(sp.block_diag([_num_csc(a) for a in args], format="csc"))
    mats = [_dense(a) for a in args]
    r = sum(x.shape[0] for x in mats)
    c = sum(x.shape[1] for x in mats)
    out = np.zeros((r, c), dtype=complex if any(x.dtype.kind == "c" for x in mats) else float)
    i = j = 0
    for x in mats:
        out[i:i + x.shape[0], j:j + x.shape[1]] = x
        i += x.shape[0]
        j += x.shape[1]
    return out


# -- native versions of general functions ------------------------------------------------------------

def _first_sparse(args: list) -> "SparseMatrix":
    return next(a for a in args if isinstance(a, SparseMatrix))


def _axis_arg(args: list, pos: int, shape: tuple):
    """Reduction axis: 0, 1, or None for 'all'."""
    if len(args) > pos:
        a = args[pos]
        if is_text(a):
            if text(a).lower() == "all":
                return None
            raise MatlabError(f"Unknown option '{text(a)}'.")
        d = to_int(a, "dimension")
        if d < 1:
            raise MatlabError("Dimension argument must be a positive integer.")
        return d - 1
    return 0 if shape[0] != 1 else 1


def _strip_flags(args: list) -> list:
    """Drop 'omitnan'-style option words that do not change a sparse reduction."""
    return [a for a in args if not (is_text(a) and text(a).lower() in
                                    ("includenan", "includemissing", "default", "double", "native"))]


@native("nnz")
def _n_nnz(interp, args, nargout):
    return np.array([[float(args[0].m.nnz)]])


@native("find")
def _n_find(interp, args, nargout):
    s = args[0]
    coo = s.m.tocoo()
    order = np.lexsort((coo.row, coo.col))
    rows, cols, vals = coo.row[order].astype(np.int64), coo.col[order].astype(np.int64), coo.data[order]
    if len(args) > 1:
        k = to_int(args[1], "n")
        last = len(args) > 2 and text(args[2]).lower() == "last"
        sel = slice(max(rows.size - k, 0), None) if last else slice(0, k)
        rows, cols, vals = rows[sel], cols[sel], vals[sel]
    shape = (1, -1) if (s.shape[0] == 1 and s.shape[1] != 1) else (-1, 1)
    if nargout <= 1:
        return (rows + cols * s.shape[0] + 1.0).reshape(shape)
    outs = [(rows + 1.0).reshape(shape), (cols + 1.0).reshape(shape)]
    if nargout > 2:
        outs.append(vals.reshape(shape) if vals.dtype == bool else vals.astype(vals.dtype).reshape(shape))
    return tuple(outs)


@native("sum")
def _n_sum(interp, args, nargout):
    args = _strip_flags(args)
    if any(is_text(a) and text(a).lower() != "all" for a in args[1:]):
        return _fallback(interp, "sum", args, nargout)
    m = args[0].fm()
    axis = _axis_arg(args, 1, m.shape)
    if axis is None:
        return np.array([[m.sum()]]) if m.dtype.kind != "c" else simplify(np.array([[m.sum()]]))
    if axis > 1:
        return args[0]
    return _maybe_scalar(sp.csc_matrix(m.sum(axis=axis)))


@native("mean")
def _n_mean(interp, args, nargout):
    args = _strip_flags(args)
    if any(is_text(a) and text(a).lower() != "all" for a in args[1:]):
        return _fallback(interp, "mean", args, nargout)
    m = args[0].fm()
    axis = _axis_arg(args, 1, m.shape)
    if axis is None:
        return simplify(np.array([[m.sum() / max(m.shape[0] * m.shape[1], 1)]]))
    if axis > 1:
        return args[0]
    count = m.shape[axis]
    with np.errstate(all="ignore"):
        return _maybe_scalar(sp.csc_matrix(m.sum(axis=axis) / count))


def _extreme(interp, name: str, args: list, nargout: int):
    s = args[0]
    if nargout > 1 or not isinstance(s, SparseMatrix) or s.m.dtype.kind == "c" or \
            any(is_text(a) and text(a).lower() != "all" for a in args[1:]):
        return _fallback(interp, name, args, nargout)
    m = s.fm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(args) >= 2 and not (isinstance(args[1], np.ndarray) and args[1].size == 0):
            other = args[1]
            if len(args) > 2:
                return _fallback(interp, name, args, nargout)
            if isinstance(other, SparseMatrix) and other.shape == s.shape:
                return SparseMatrix(m.maximum(other.fm()) if name == "max" else m.minimum(other.fm()))
            res = interp.call_named(name, [s.to_dense(), _dense(other)], 1, interp.frame)[0]
            return SparseMatrix(sp.csc_matrix(res))
        axis = _axis_arg(args, 2, m.shape)
        if m.shape[0] * m.shape[1] == 0:
            return _fallback(interp, name, args, nargout)
        has_nan = m.nnz and np.isnan(m.data).any()
        if has_nan:
            return _fallback(interp, name, args, nargout)
        if axis is None:
            return np.array([[float(m.max() if name == "max" else m.min())]])
        if axis > 1:
            return s
        res = m.max(axis=axis) if name == "max" else m.min(axis=axis)
        return _maybe_scalar(sp.csc_matrix(res))


@native("max")
def _n_max(interp, args, nargout):
    return _extreme(interp, "max", args, nargout)


@native("min")
def _n_min(interp, args, nargout):
    return _extreme(interp, "min", args, nargout)


def _any_all(interp, name: str, args: list, nargout: int):
    m = args[0].m
    axis = _axis_arg(args, 1, m.shape)
    nz = (m != 0) if m.dtype != bool else m
    if axis is None:
        total = m.shape[0] * m.shape[1]
        return bool_scalar(nz.nnz > 0 if name == "any" else nz.nnz == total)
    if axis > 1:
        return SparseMatrix(nz)
    counts = np.asarray(nz.astype(np.int64).sum(axis=axis))
    flags = counts > 0 if name == "any" else counts == m.shape[axis]
    if flags.shape == (1, 1):
        return flags
    return SparseMatrix(sp.csc_matrix(flags))


@native("any")
def _n_any(interp, args, nargout):
    return _any_all(interp, "any", args, nargout)


@native("all")
def _n_all(interp, args, nargout):
    return _any_all(interp, "all", args, nargout)


def _fallback(interp, name: str, args: list, nargout: int):
    dense = [a.to_dense() if isinstance(a, SparseMatrix) else a for a in args]
    res = interp.call_named(name, dense, nargout, interp.frame)
    return tuple(res) if len(res) != 1 else res[0]


_ZERO_PRESERVING = ["abs", "sqrt", "sin", "tan", "asin", "atan", "sinh", "tanh", "asinh", "atanh", "floor",
                    "ceil", "round", "fix", "sign", "real", "imag", "conj", "expm1", "log1p", "deg2rad",
                    "rad2deg", "sind", "tand", "asind", "atand", "nthroot", "realsqrt", "cbrt"]


def _make_zero_preserving(name: str):
    def f(interp, args, nargout):
        s = args[0]
        if not isinstance(s, SparseMatrix) or any(isinstance(a, SparseMatrix) for a in args[1:]):
            return _fallback(interp, name, args, nargout)
        m = s.fm()
        if m.nnz == 0:
            return SparseMatrix(m)
        res = interp.call_named(name, [m.data.reshape(-1, 1).copy(), *args[1:]], 1, interp.frame)[0]
        return _with_data(m, to_numeric(res).flatten(order="F"))
    return f


for _name in _ZERO_PRESERVING:
    if _name in BUILTINS:
        _NATIVE[_name] = _make_zero_preserving(_name)


@native("double")
def _n_double(interp, args, nargout):
    return SparseMatrix(args[0].fm())


@native("logical")
def _n_logical(interp, args, nargout):
    m = args[0].m
    if m.dtype.kind in "fc" and m.nnz and np.isnan(m.data).any():
        raise MatlabError("NaN's cannot be converted to logicals.")
    return SparseMatrix(m.astype(bool))


@native("isnan")
def _n_isnan(interp, args, nargout):
    m = args[0].fm()
    return _with_data(m, np.isnan(m.data))


@native("isinf")
def _n_isinf(interp, args, nargout):
    m = args[0].fm()
    return _with_data(m, np.isinf(m.data))


@native("diag")
def _n_diag(interp, args, nargout):
    m = args[0].m
    k = to_int(args[1], "diagonal") if len(args) > 1 else 0
    if 1 in m.shape:
        v = np.asarray(m.toarray()).ravel()
        n = v.size + abs(k)
        idx = np.nonzero(v)[0]
        rows = idx + max(-k, 0)
        cols = idx + max(k, 0)
        return SparseMatrix(sp.csc_matrix((v[idx], (rows, cols)), shape=(n, n)))
    return SparseMatrix(sp.csc_matrix(m.diagonal(k).reshape(-1, 1)))


@native("tril")
def _n_tril(interp, args, nargout):
    return SparseMatrix(sp.tril(args[0].m, to_int(args[1], "diagonal") if len(args) > 1 else 0))


@native("triu")
def _n_triu(interp, args, nargout):
    return SparseMatrix(sp.triu(args[0].m, to_int(args[1], "diagonal") if len(args) > 1 else 0))


@native("transpose")
def _n_transpose(interp, args, nargout):
    return SparseMatrix(args[0].m.T)


@native("ctranspose")
def _n_ctranspose(interp, args, nargout):
    return SparseMatrix(args[0].m.conj().T)


def _op_function(op: str):
    def f(interp, args, nargout):
        if len(args) != 2:
            raise MatlabError("This operator function needs two inputs.")
        return _binop(op, args[0], args[1], interp)
    return f


for _op, _fname in {"+": "plus", "-": "minus", ".*": "times", "*": "mtimes", "./": "rdivide", "/": "mrdivide",
                    ".\\": "ldivide", "\\": "mldivide", ".^": "power", "^": "mpower", "==": "eq", "~=": "ne",
                    "<": "lt", "<=": "le", ">": "gt", ">=": "ge", "&": "and", "|": "or"}.items():
    _NATIVE[_fname] = _op_function(_op)


@native("uminus")
def _n_uminus(interp, args, nargout):
    return args[0].unary_op(interp, "-")


@native("uplus")
def _n_uplus(interp, args, nargout):
    return args[0].unary_op(interp, "+")


@native("not")
def _n_not(interp, args, nargout):
    return args[0].unary_op(interp, "~")


@native("kron")
def _n_kron(interp, args, nargout):
    return SparseMatrix(sp.kron(_num_csc(args[0]), _num_csc(args[1]), format="csc"))


@native("trace")
def _n_trace(interp, args, nargout):
    m = args[0].fm()
    if m.shape[0] != m.shape[1]:
        raise MatlabError("Matrix must be square.")
    return simplify(np.array([[m.diagonal().sum()]]))


@native("isequal")
def _n_isequal(interp, args, nargout):
    return bool_scalar(all(_equal(args[0], b) for b in args[1:]))


def _two_norm(m: sp.csc_matrix) -> float:
    if m.nnz == 0:
        return 0.0
    if min(m.shape) <= 2 or m.shape[0] * m.shape[1] <= 250_000:
        return float(np.linalg.norm(m.toarray(), 2))
    return float(spla.svds(m.astype(float) if m.dtype.kind != "c" else m, k=1, return_singular_vectors=False)[0])


@native("norm")
def _n_norm(interp, args, nargout):
    m = args[0].fm()
    p: Any = 2
    if len(args) > 1:
        p = text(args[1]).lower() if is_text(args[1]) else float(np.real(to_numeric(args[1]).flat[0]))
    data = np.abs(m.data)
    if 1 in m.shape or 0 in m.shape:  # vector norms
        if p == "fro" or p == 2:
            return float(np.sqrt(np.sum(data ** 2)))
        if p == "inf" or p == np.inf:
            return float(data.max()) if data.size else 0.0
        if p == -np.inf:
            return float(data.min()) if data.size == m.shape[0] * m.shape[1] and data.size else 0.0
        if isinstance(p, str):
            raise MatlabError(f"Unknown norm type '{p}'.")
        if p == 0:
            raise MatlabError("The norm order must be nonzero.")
        return float(np.sum(data ** p) ** (1.0 / p))
    am = abs(m)
    if p == 1:
        return float(am.sum(axis=0).max())
    if p == "inf" or p == np.inf:
        return float(am.sum(axis=1).max())
    if p == "fro":
        return float(np.sqrt(np.sum(data ** 2)))
    if p == 2:
        return _two_norm(m)
    raise MatlabError("The only matrix norms available are 1, 2, inf, and 'fro'.")


@sbuiltin("normest")
def _normest(interp, args, nargout):
    return _two_norm(_num_csc(args[0]))


@sbuiltin("condest")
def _condest(interp, args, nargout):
    m = _num_csc(args[0])
    if m.shape[0] != m.shape[1]:
        raise MatlabError("condest: the matrix must be square.")
    if m.shape[0] == 0:
        return 0.0
    norm_a = float(abs(m).sum(axis=0).max())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            lu = spla.splu(m)
    except (RuntimeError, spla.MatrixRankWarning):
        return float("inf")
    n = m.shape[0]
    op = spla.LinearOperator((n, n), matvec=lu.solve, rmatvec=lambda x: lu.solve(x, trans="H"), dtype=m.dtype)
    return norm_a * float(spla.onenormest(op))


@native("inv")
def _n_inv(interp, args, nargout):
    s = args[0]
    if s.shape[0] != s.shape[1]:
        raise MatlabError("Matrix must be square.")
    return _binop("\\", s, SparseMatrix(sp.eye(s.shape[0], format="csc")), interp)


@native("repmat")
def _n_repmat(interp, args, nargout):
    s = args[0]
    if not isinstance(s, SparseMatrix):
        return _fallback(interp, "repmat", args, nargout)
    r, c = _dims(args[1:], "repmat")
    return SparseMatrix(sp.kron(sp.csc_matrix(np.ones((max(r, 0), max(c, 0)))), s.m, format="csc")
                        if not s.is_logical else
                        sp.kron(sp.csc_matrix(np.ones((max(r, 0), max(c, 0)))), s.fm(), format="csc").astype(bool))


@native("reshape")
def _n_reshape(interp, args, nargout):
    s = args[0]
    if not isinstance(s, SparseMatrix):
        return _fallback(interp, "reshape", args, nargout)
    total = s.size
    if len(args) == 2:
        dims = [int(x) for x in _dense(args[1]).flatten()]
    else:
        dims = [None if (isinstance(a, np.ndarray) and a.size == 0) else to_int(a, "size") for a in args[1:]]
    if dims.count(None) > 1:
        raise MatlabError("reshape: only one size may be left as [].")
    if None in dims:
        known = int(np.prod([d for d in dims if d is not None]))
        if known == 0 or total % known:
            raise MatlabError("reshape: the number of elements must be divisible by the known sizes.")
        dims[dims.index(None)] = total // known
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    if len(dims) != 2:
        raise MatlabError("A sparse matrix can only be reshaped to two dimensions.")
    if dims[0] * dims[1] != total:
        raise MatlabError("Number of elements must not change. Use [] as one of the size inputs to "
                          "automatically calculate the appropriate size for that dimension.",
                          "MATLAB:getReshapeDims:notSameNumel")
    return SparseMatrix(_reshape(s.m, (dims[0], dims[1])))


def _cat(axis_from_args: bool, axis: int = 0):
    def f(interp, args, nargout):
        ax, items = axis, list(args)
        if axis_from_args:
            d = to_int(items[0], "dimension")
            if d not in (1, 2):
                raise MatlabError("Sparse matrices can only be concatenated along dimension 1 or 2.")
            ax, items = (0 if d == 1 else 1), items[1:]
        return SparseMatrix.concat(items, ax)
    return f


_NATIVE["cat"] = _cat(True)
_NATIVE["horzcat"] = _cat(False, 1)
_NATIVE["vertcat"] = _cat(False, 0)


def _structure(name: str):
    def f(interp, args, nargout):
        m = args[0].m
        coo = m.tocoo()
        i, j = coo.row.astype(np.int64), coo.col.astype(np.int64)
        if name == "isdiag":
            return bool_scalar(bool(np.all(i == j)))
        if name == "istril":
            return bool_scalar(bool(np.all(i >= j)))
        if name == "istriu":
            return bool_scalar(bool(np.all(i <= j)))
        if m.shape[0] != m.shape[1]:
            return bool_scalar(False)
        other = m.T if name == "issymmetric" else m.conj().T
        if len(args) > 1 and is_text(args[1]) and text(args[1]).lower() == "skew":
            other = -other
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bool_scalar((m.astype(other.dtype) != other).nnz == 0 if m.dtype != bool else (m != other).nnz == 0)
    return f


for _name in ("isdiag", "istril", "istriu", "issymmetric", "ishermitian"):
    _NATIVE[_name] = _structure(_name)


@sbuiltin("bandwidth")
def _bandwidth(interp, args, nargout):
    coo = as_csc(args[0]).tocoo()
    d = coo.col.astype(np.int64) - coo.row
    lower = float(max(0, -d.min())) if d.size else 0.0
    upper = float(max(0, d.max())) if d.size else 0.0
    if len(args) > 1:
        return lower if text(args[1]).lower() == "lower" else upper
    return (lower, upper) if nargout > 1 else lower


@sbuiltin("isbanded")
def _isbanded(interp, args, nargout):
    lower, upper = _bandwidth(interp, [args[0]], 2)
    return bool_scalar(lower <= float(np.real(to_numeric(args[1]).flat[0]))
                       and upper <= float(np.real(to_numeric(args[2]).flat[0])))


# -- factorizations ----------------------------------------------------------------------------------

def _perm_matrix(perm: np.ndarray, rows: bool) -> sp.csc_matrix:
    n = perm.size
    ones = np.ones(n)
    if rows:  # P with P*A = A(perm,:)
        return sp.csc_matrix((ones, (np.arange(n), perm)), shape=(n, n))
    return sp.csc_matrix((ones, (perm, np.arange(n))), shape=(n, n))


@native("lu")
def _n_lu(interp, args, nargout):
    s = args[0]
    if s.shape[0] != s.shape[1] or s.shape[0] == 0 or len(args) > 1:
        return _fallback(interp, "lu", args, nargout)
    m = s.fm()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = spla.splu(m, permc_spec="COLAMD" if nargout >= 4 else "NATURAL")
    except RuntimeError:
        return _fallback(interp, "lu", args, nargout)
    n = m.shape[0]
    # SuperLU: L*U = Pr*A*Pc with Pr[perm_r[i], i] = 1 and Pc[i, perm_c[i]] = 1
    row_order = np.empty(n, dtype=np.int64)
    row_order[f.perm_r] = np.arange(n)  # row_order[k] = original row placed at position k
    P = _perm_matrix(row_order, rows=True)
    L, U = SparseMatrix(f.L), SparseMatrix(f.U)
    if nargout <= 2:
        return SparseMatrix(P.T @ f.L), U
    if nargout == 3:
        return L, U, SparseMatrix(P)
    Q = _perm_matrix(f.perm_c, rows=False)
    return L, U, SparseMatrix(P), SparseMatrix(Q)


@native("chol")
def _n_chol(interp, args, nargout):
    s = args[0]
    lower = any(is_text(a) and text(a).lower() == "lower" for a in args[1:])
    if s.shape[0] != s.shape[1]:
        raise MatlabError("Matrix must be square.")
    n = s.shape[0]
    m = s.fm()
    tri = sp.triu(m) if not lower else sp.tril(m)
    sym = (tri + sp.triu(m, 1).conj().T if not lower else tri + sp.tril(m, -1).conj().T).tocsc()
    ok = n > 0
    R = None
    if ok:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = spla.splu(sym, permc_spec="NATURAL", diag_pivot_thresh=0.0,
                              options={"SymmetricMode": True})
            d = f.U.diagonal()
            ok = bool(np.all(f.perm_r == np.arange(n)) and np.all(np.real(d) > 0)
                      and np.all(np.abs(np.imag(d)) <= 1e-12 * np.abs(d)))
            if ok:
                R = sp.diags(1.0 / np.sqrt(np.real(d))) @ f.U
        except RuntimeError:
            ok = False
    if n == 0:
        ok, R = True, sp.csc_matrix((0, 0))
    if not ok:
        if nargout > 1:
            return SparseMatrix(sp.csc_matrix((0, 0))), np.array([[1.0]])
        raise MatlabError("Matrix must be positive definite.", "MATLAB:posdef")
    out = SparseMatrix(R.conj().T if lower else R)
    return (out, np.array([[0.0]])) if nargout > 1 else out


def _ilu_factors(m: sp.csc_matrix, drop_tol: float, fill: float):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = spla.spilu(m, drop_tol=drop_tol, fill_factor=fill, permc_spec="NATURAL", diag_pivot_thresh=0.0)
    n = m.shape[0]
    row_order = np.empty(n, dtype=np.int64)
    row_order[f.perm_r] = np.arange(n)
    return f.L.tocsc(), f.U.tocsc(), _perm_matrix(row_order, rows=True)


def _ilu0(m: sp.csc_matrix) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """Incomplete LU with no fill-in: L (unit lower) and U keep the nonzero pattern of the matrix."""
    csr = m.tocsr()
    csr.sort_indices()
    n = csr.shape[0]
    indptr, indices = csr.indptr, csr.indices
    data = csr.data.astype(complex if csr.dtype.kind == "c" else float)
    diag_pos = np.full(n, -1, dtype=np.int64)
    upper: list[dict] = [{} for _ in range(n)]  # finished rows of U beyond the diagonal: {column: position}
    for i in range(n):
        lo, hi = indptr[i], indptr[i + 1]
        cols = indices[lo:hi]
        where = {int(c): lo + t for t, c in enumerate(cols)}
        for t in range(lo, hi):
            k = int(indices[t])
            if k >= i:
                break
            pivot = data[diag_pos[k]] if diag_pos[k] >= 0 else 0.0
            if pivot == 0:
                raise MatlabError("ilu: encountered a zero pivot.", "MATLAB:ilu:ZeroPivot")
            factor = data[t] / pivot
            data[t] = factor
            for j, pos in upper[k].items():
                hit = where.get(j)
                if hit is not None:
                    data[hit] -= factor * data[pos]
        if i not in where or data[where[i]] == 0:
            raise MatlabError("ilu: encountered a zero pivot.", "MATLAB:ilu:ZeroPivot")
        diag_pos[i] = where[i]
        upper[i] = {c: pos for c, pos in where.items() if c > i}
    full = sp.csr_matrix((data, indices.copy(), indptr.copy()), shape=csr.shape)
    return (sp.tril(full, -1) + sp.eye(n)).tocsc(), sp.triu(full).tocsc()


def _ichol0(low: sp.csc_matrix) -> sp.csc_matrix:
    """Incomplete Cholesky with no fill-in, for the lower triangle ``low`` of a symmetric matrix."""
    csr = low.tocsr()
    csr.sort_indices()
    n = csr.shape[0]
    indptr, indices = csr.indptr.tolist(), csr.indices.tolist()
    data = csr.data.astype(complex if csr.dtype.kind == "c" else float)
    is_complex = data.dtype.kind == "c"
    data = data.tolist() if not is_complex else data
    rows: list[dict] = [{} for _ in range(n)]  # finished rows below the diagonal: {column: value}
    diag = [0.0] * n
    for i in range(n):
        lo, hi = indptr[i], indptr[i + 1]
        current: dict = {}
        acc = 0.0
        d = None
        for t in range(lo, hi):
            k = int(indices[t])
            if k == i:
                d = data[t]
                break
            rk = rows[k]
            small, big = (current, rk) if len(current) < len(rk) else (rk, current)
            if not small:
                dot = 0.0
            elif is_complex:
                dot = sum(v * np.conj(big[c]) for c, v in small.items() if c in big)
                dot = dot if small is current else np.conj(dot)
            else:
                dot = sum(v * big[c] for c, v in small.items() if c in big)
            val = (data[t] - dot) / diag[k]
            data[t] = val
            current[k] = val
            acc += abs(val) ** 2
        pivot = (complex(d).real if d is not None else 0.0) - acc
        if d is None or not pivot > 0:
            raise MatlabError("ichol: encountered a nonpositive pivot.", "MATLAB:ichol:Breakdown")
        diag[i] = float(np.sqrt(pivot))
        data[hi - 1] = diag[i]
        rows[i] = current
    return sp.csr_matrix((np.asarray(data), csr.indices.copy(), csr.indptr.copy()), shape=csr.shape).tocsc()


def _setup_value(opts: Any, name: str, default: float) -> float:
    if isinstance(opts, Struct) and name in opts.fields:
        return float(np.real(to_numeric(opts.get_field(0, name)).flat[0]))
    return default


def _setup_type(opts: Any, default: str) -> str:
    if isinstance(opts, Struct) and "type" in opts.fields:
        return to_str(opts.get_field(0, "type")).lower()
    return default


@sbuiltin("ilu")
def _ilu(interp, args, nargout):
    m = _num_csc(args[0])
    if m.shape[0] != m.shape[1]:
        raise MatlabError("ilu: the matrix must be square.")
    opts = args[1] if len(args) > 1 else None
    kind = _setup_type(opts, "nofill")
    drop = _setup_value(opts, "droptol", 0.0)
    try:
        if kind == "nofill":
            L, U = _ilu0(m)
            P = sp.eye(m.shape[0], format="csc")
        else:
            L, U, P = _ilu_factors(m, drop, 10.0)
    except RuntimeError:
        raise MatlabError("ilu: the factorization hit a zero pivot.") from None
    if nargout >= 3:
        return SparseMatrix(L), SparseMatrix(U), SparseMatrix(P)
    if nargout == 2:
        return SparseMatrix(P.T @ L), SparseMatrix(U)
    return SparseMatrix(P.T @ (L - sp.eye(m.shape[0])) + U)


@sbuiltin("ichol")
def _ichol(interp, args, nargout):
    m = _num_csc(args[0])
    n = m.shape[0]
    if m.shape[0] != m.shape[1]:
        raise MatlabError("ichol: the matrix must be square.")
    opts = args[1] if len(args) > 1 else None
    kind = _setup_type(opts, "nofill")
    drop = _setup_value(opts, "droptol", 0.0)
    shift = _setup_value(opts, "diagcomp", 0.0)
    low = sp.tril(m)
    sym = (low + sp.tril(m, -1).conj().T).tocsc()
    if shift:
        sym = sym + shift * sp.diags(sym.diagonal())
    if kind == "nofill":
        return SparseMatrix(_ichol0(sp.tril(sym).tocsc()))
    try:
        L, U, P = _ilu_factors(sym.tocsc(), drop, 10.0)
    except RuntimeError:
        raise MatlabError("ichol: encountered a nonpositive pivot.", "MATLAB:ichol:Breakdown") from None
    d = np.real(U.diagonal())
    if np.any(d <= 0) or (P != sp.eye(n)).nnz:
        raise MatlabError("ichol: encountered a nonpositive pivot.", "MATLAB:ichol:Breakdown")
    return SparseMatrix(sp.tril(L @ sp.diags(np.sqrt(d))))


# -- orderings -----------------------------------------------------------------------------------------

@sbuiltin("symrcm")
def _symrcm(interp, args, nargout):
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    m = _num_csc(args[0])
    pat = (abs(m) + abs(m).T).tocsr()
    return (reverse_cuthill_mckee(pat, symmetric_mode=True) + 1.0).reshape(1, -1)


def _ordering(spec: str, name: str):
    def f(interp, args, nargout):
        m = _num_csc(args[0])
        n = m.shape[1]
        if n == 0:
            return np.zeros((1, 0))
        pat = _with_data(m, np.ones(m.nnz)).m
        if m.shape[0] == m.shape[1]:
            work = (pat + pat.T + 2 * n * sp.eye(n)).tocsc() if name != "colamd" else (pat + 2 * n * sp.eye(n)).tocsc()
        else:
            work = (pat.T @ pat + 2 * n * sp.eye(n)).tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = spla.splu(work, permc_spec=spec).perm_c
        return (np.asarray(perm) + 1.0).reshape(1, -1)
    return f


sbuiltin("amd", "symamd")(_ordering("MMD_AT_PLUS_A", "amd"))
sbuiltin("colamd")(_ordering("COLAMD", "colamd"))


@sbuiltin("sprank")
def _sprank(interp, args, nargout):
    from scipy.sparse.csgraph import structural_rank
    return float(structural_rank(_num_csc(args[0]).tocsr()))


# -- eigenvalues and singular values --------------------------------------------------------------------

_WHICH = {"lm": "LM", "largestabs": "LM", "sm": "SM", "smallestabs": "SM", "la": "LA", "largestreal": "LR",
          "sa": "SA", "smallestreal": "SR", "lr": "LR", "sr": "SR", "be": "BE", "bothendsreal": "BE",
          "li": "LI", "si": "SI", "largestimag": "LI", "smallestimag": "SI"}


def _eig_order(vals: np.ndarray, which: str) -> np.ndarray:
    if which == "LM":
        return np.argsort(-np.abs(vals), kind="stable")
    if which == "SM":
        return np.argsort(np.abs(vals), kind="stable")
    if which in ("LA", "LR"):
        return np.argsort(-np.real(vals), kind="stable")
    if which in ("SA", "SR"):
        return np.argsort(np.real(vals), kind="stable")
    if which == "LI":
        return np.argsort(-np.imag(vals), kind="stable")
    if which == "SI":
        return np.argsort(np.imag(vals), kind="stable")
    return np.argsort(np.real(vals), kind="stable")


@sbuiltin("eigs")
def _eigs(interp, args, nargout):
    if not args or isinstance(args[0], FunctionHandle):
        raise MatlabError("eigs: pass the matrix itself; function-handle operators are not supported.")
    a = _num_csc(args[0])
    n = a.shape[0]
    if a.shape[0] != a.shape[1]:
        raise MatlabError("eigs: the matrix must be square.")
    rest = list(args[1:])
    b = None
    if rest and (isinstance(rest[0], SparseMatrix) or (isinstance(rest[0], np.ndarray) and not is_char(rest[0])
                                                       and rest[0].shape == a.shape and n > 1)):
        b = _num_csc(rest.pop(0))
    k = 6
    if rest and not is_text(rest[0]):
        if not (isinstance(rest[0], np.ndarray) and rest[0].size == 0):
            k = to_int(rest[0], "k")
        rest.pop(0)
    k = max(0, min(k, n))
    which, sigma = "LM", None
    if rest:
        s = rest.pop(0)
        if is_text(s):
            key = text(s).lower()
            if key not in _WHICH:
                raise MatlabError(f"eigs: unknown option '{text(s)}'. Use 'largestabs', 'smallestabs', "
                                  "'largestreal', 'smallestreal', 'bothendsreal' or a numeric shift.")
            which = _WHICH[key]
        elif isinstance(s, Struct):
            pass
        else:
            sigma = complex(to_numeric(s).flat[0])
            sigma = sigma.real if sigma.imag == 0 else sigma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        herm = (a != a.conj().T).nnz == 0 and (b is None or (b != b.conj().T).nnz == 0)
    if k == 0:
        vals, vecs = np.zeros(0), np.zeros((n, 0))
    elif n <= 400 or k >= n - 1 - (0 if herm else 1):
        import scipy.linalg as sla
        ad = a.toarray()
        bd = None if b is None else b.toarray()
        if herm:
            vals, vecs = sla.eigh(ad, bd)
        else:
            vals, vecs = sla.eig(ad, bd)
        if sigma is not None:
            order = np.argsort(np.abs(vals - sigma), kind="stable")[:k]
            order = order[np.argsort(-np.abs(vals[order] - sigma), kind="stable")][::-1]
        elif which == "BE":
            asc = np.argsort(np.real(vals), kind="stable")
            order = np.concatenate([asc[:k // 2], asc[len(asc) - (k - k // 2):]])
        else:
            order = _eig_order(vals, which)[:k]
        vals, vecs = vals[order], vecs[:, order]
    else:
        kw: dict = {"k": k, "M": b}
        solver = spla.eigsh if herm else spla.eigs
        w = which
        if sigma is not None or which == "SM":
            kw["sigma"] = 0.0 if sigma is None else sigma
            w = "LM"
        elif herm:
            w = {"LR": "LA", "SR": "SA"}.get(which, which)
            if w in ("LI", "SI"):
                w = "LM"
        else:
            w = {"LA": "LR", "SA": "SR", "BE": "LR"}.get(which, which)
        try:
            vals, vecs = solver(a.astype(complex) if (not herm and a.dtype.kind == "c") else a, which=w, **kw)
        except spla.ArpackNoConvergence as e:
            raise MatlabError(f"eigs: only {len(e.eigenvalues)} of the {k} requested eigenvalues converged.") from None
        except RuntimeError as e:  # singular shift
            raise MatlabError(f"eigs: the shifted matrix is singular; choose a different shift ({e}).") from None
        if sigma is not None:
            order = np.argsort(np.abs(vals - sigma), kind="stable")
        elif which == "BE":
            order = np.argsort(np.real(vals), kind="stable")
        else:
            order = _eig_order(vals, which)
        vals, vecs = vals[order], vecs[:, order]
    vals = simplify(np.asarray(vals).reshape(-1, 1))
    if nargout <= 1:
        return vals
    d = np.zeros((k, k), dtype=vals.dtype)
    d[np.arange(k), np.arange(k)] = vals.ravel()
    outs = (simplify(np.asarray(vecs)), d, np.array([[0.0]]))
    return outs[:max(nargout, 2)]


@sbuiltin("svds")
def _svds(interp, args, nargout):
    a = _num_csc(args[0])
    r, c = a.shape
    k = 6
    rest = list(args[1:])
    if rest and not is_text(rest[0]):
        k = to_int(rest.pop(0), "k")
    k = max(0, min(k, r, c))
    smallest = False
    if rest:
        s = rest.pop(0)
        if is_text(s):
            key = text(s).lower()
            if key in ("smallest", "smallestnz"):
                smallest = True
            elif key != "largest":
                raise MatlabError(f"svds: unknown option '{text(s)}'. Use 'largest' or 'smallest'.")
        else:
            smallest = float(np.real(to_numeric(s).flat[0])) == 0
            if not smallest:
                raise MatlabError("svds: only a shift of 0 (the smallest singular values) is supported.")
    if k == 0:
        u, sv, vt = np.zeros((r, 0)), np.zeros(0), np.zeros((0, c))
    elif min(r, c) <= 400 or k >= min(r, c) - 1 or smallest and r * c <= 4_000_000:
        u, sv, vt = np.linalg.svd(a.toarray(), full_matrices=False)
        sel = np.arange(sv.size - k, sv.size)[::-1] if smallest else np.arange(k)
        if smallest:
            sel = sel[::-1][np.argsort(sv[sel[::-1]], kind="stable")][::-1]
            sel = np.sort(sel)  # descending values, as for the largest case
        u, sv, vt = u[:, sel], sv[sel], vt[sel, :]
    else:
        try:
            u, sv, vt = spla.svds(a, k=k, which="SM" if smallest else "LM")
        except spla.ArpackNoConvergence:
            raise MatlabError("svds: the iteration did not converge.") from None
        order = np.argsort(-sv, kind="stable")
        u, sv, vt = u[:, order], sv[order], vt[order, :]
    if nargout <= 1:
        return sv.reshape(-1, 1)
    outs = (simplify(u), np.diag(sv) if sv.size else np.zeros((0, 0)), simplify(vt.conj().T), np.array([[0.0]]))
    return outs[:max(nargout, 3)]


# -- iterative solvers ----------------------------------------------------------------------------------

def _operator(interp, a: Any, n: int, what: str):
    """A callable x -> A*x for a matrix (sparse or full) or a function handle."""
    if isinstance(a, FunctionHandle) or is_text(a):
        def apply(x):
            res = interp.call_function_value(a, [np.asarray(x).reshape(-1, 1)], 1)
            return _dense(res[0]).reshape(-1)
        return apply, None
    m = _num_csc(a) if isinstance(a, SparseMatrix) else _dense(a)
    if m.shape != (n, n):
        raise MatlabError(f"{what} must be a square matrix whose size matches the right-hand side.")
    return (lambda x: np.asarray(m @ x).reshape(-1)), m


def _precond_solver(interp, mval: Any, n: int):
    """A callable x -> M \\ x, or None when no preconditioner is given."""
    if mval is None or (isinstance(mval, np.ndarray) and mval.size == 0):
        return None
    if isinstance(mval, FunctionHandle):
        def apply(x):
            res = interp.call_function_value(mval, [np.asarray(x).reshape(-1, 1)], 1)
            return _dense(res[0]).reshape(-1)
        return apply
    if isinstance(mval, SparseMatrix):
        m = mval.fm()
        if m.shape != (n, n):
            raise MatlabError("The preconditioner must be a square matrix of the same size as A.")
        coo = m.tocoo()
        if np.all(coo.row >= coo.col) or np.all(coo.row <= coo.col):
            lower = bool(np.all(coo.row >= coo.col))
            csr = m.tocsr()
            return lambda x: spla.spsolve_triangular(csr, np.asarray(x), lower=lower)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lu = spla.splu(m)
        return lu.solve
    import scipy.linalg as sla
    d = _dense(mval)
    if d.shape != (n, n):
        raise MatlabError("The preconditioner must be a square matrix of the same size as A.")
    fac = sla.lu_factor(d)
    return lambda x: sla.lu_solve(fac, x)


def _opt(args: list, pos: int):
    if len(args) > pos and not (isinstance(args[pos], np.ndarray) and args[pos].size == 0):
        return args[pos]
    return None


def _krylov(method: str):
    def f(interp, args, nargout):
        if len(args) < 2:
            raise MatlabError(f"Not enough input arguments for {method}.")
        b = _dense(args[1])
        if b.ndim != 2 or b.shape[1] != 1:
            raise MatlabError(f"{method}: the right-hand side must be a column vector.")
        n = b.shape[0]
        matvec, mat = _operator(interp, args[0], n, "A")
        rest = list(args[2:])
        restart = None
        if method == "gmres":
            r = _opt(rest, 0)
            restart = None if r is None else to_int(r, "restart")
            rest = rest[1:]
        tol_v, maxit_v = _opt(rest, 0), _opt(rest, 1)
        tol = 1e-6 if tol_v is None else float(np.real(to_numeric(tol_v).flat[0]))
        if method == "gmres":
            inner = n if restart is None else min(restart, n)
            outer = (1 if restart is None else min(max(n // max(inner, 1), 1), 10)) if maxit_v is None \
                else to_int(maxit_v, "maxit")
            if restart is None and maxit_v is None:
                inner, outer = min(n, 10), 1
            elif restart is None:
                inner, outer = min(to_int(maxit_v, "maxit"), n), 1
            maxit = outer
        else:
            maxit = min(n, 20) if maxit_v is None else to_int(maxit_v, "maxit")
        m1 = _precond_solver(interp, _opt(rest, 2), n)
        m2 = _precond_solver(interp, _opt(rest, 3), n)
        x0v = _opt(rest, 4)
        x0 = None if x0v is None else _dense(x0v).reshape(-1).astype(b.dtype if b.dtype.kind == "c" else float)
        complex_problem = b.dtype.kind == "c" or (mat is not None and mat.dtype.kind == "c")
        dtype = complex if complex_problem else float
        A = spla.LinearOperator((n, n), matvec=matvec, dtype=dtype)
        M = None
        if m1 is not None or m2 is not None:
            def apply_m(x):
                y = np.asarray(x).reshape(-1)
                if m1 is not None:
                    y = m1(y)
                if m2 is not None:
                    y = m2(y)
                return np.asarray(y).reshape(-1)
            M = spla.LinearOperator((n, n), matvec=apply_m, dtype=dtype)
        rhs = b.reshape(-1).astype(dtype)
        bnorm = float(np.linalg.norm(rhs))
        count = [0]
        resvec: list[float] = []
        want_res = nargout >= 5

        def cb(xk):
            count[0] += 1
            if want_res:
                if np.ndim(xk) == 0:
                    resvec.append(float(xk) * (bnorm if bnorm else 1.0))
                else:
                    resvec.append(float(np.linalg.norm(rhs - matvec(xk))))

        if bnorm == 0:
            x, info = np.zeros(n, dtype=dtype), 0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if method == "pcg":
                    x, info = spla.cg(A, rhs, x0=x0, rtol=tol, atol=0.0, maxiter=maxit, M=M, callback=cb)
                elif method == "bicgstab":
                    x, info = spla.bicgstab(A, rhs, x0=x0, rtol=tol, atol=0.0, maxiter=maxit, M=M, callback=cb)
                elif method == "cgs":
                    x, info = spla.cgs(A, rhs, x0=x0, rtol=tol, atol=0.0, maxiter=maxit, M=M, callback=cb)
                elif method == "minres":
                    x, info = spla.minres(A, rhs, x0=x0, rtol=tol, maxiter=maxit, M=M, callback=cb)
                elif method == "bicg":
                    if mat is None:
                        raise MatlabError("bicg needs the matrix itself, not a function handle.")
                    op = spla.aslinearoperator(mat.astype(dtype))
                    x, info = spla.bicg(op, rhs, x0=x0, rtol=tol, atol=0.0, maxiter=maxit, M=None, callback=cb)
                else:  # gmres
                    x, info = spla.gmres(A, rhs, x0=x0, rtol=tol, atol=0.0, restart=inner, maxiter=outer, M=M,
                                         callback=cb, callback_type="pr_norm")
        x = np.asarray(x).reshape(-1, 1)
        relres = float(np.linalg.norm(rhs - matvec(x.reshape(-1))) / bnorm) if bnorm else 0.0
        flag = 0.0 if relres <= tol * (1 + 1e-9) or info == 0 else (1.0 if info > 0 else 4.0)
        if info == 0 and relres > tol * 10:  # converged in the preconditioned norm only
            flag = 0.0
        its = count[0]
        if method == "gmres":
            it_out = np.array([[float((its - 1) // inner + 1 if its else 0),
                                float((its - 1) % inner + 1 if its else 0)]])
        else:
            it_out = np.array([[float(its)]])
        x = simplify(x)
        if nargout <= 1:
            shown = int(its)
            if flag == 0:
                interp.out(f"{method} converged at iteration {shown} to a solution with relative residual "
                           f"{relres:.2g}.\n")
            else:
                interp.out(f"{method} stopped at iteration {shown} without converging to the desired tolerance "
                           f"{tol:.2g} because the maximum number of iterations was reached.\n"
                           f"The iterate returned has relative residual {relres:.2g}.\n")
            return x
        outs = (x, np.array([[flag]]), np.array([[relres]]), it_out,
                np.array(resvec, dtype=float).reshape(-1, 1))
        return outs[:nargout]
    return f


for _name in ("pcg", "gmres", "bicgstab", "cgs", "minres", "bicg"):
    sbuiltin(_name)(_krylov(_name))


@sbuiltin("lsqr")
def _lsqr(interp, args, nargout):
    a = _num_csc(args[0]) if isinstance(args[0], SparseMatrix) else _dense(args[0])
    b = _dense(args[1]).reshape(-1)
    tol_v, maxit_v = _opt(args, 2), _opt(args, 3)
    tol = 1e-6 if tol_v is None else float(np.real(to_numeric(tol_v).flat[0]))
    maxit = min(a.shape[0], a.shape[1], 20) if maxit_v is None else to_int(maxit_v, "maxit")
    res = spla.lsqr(a, b, atol=tol, btol=tol, iter_lim=maxit)
    x = res[0].reshape(-1, 1)
    bnorm = float(np.linalg.norm(b))
    relres = float(np.linalg.norm(b - np.asarray(a @ res[0]).reshape(-1)) / bnorm) if bnorm else 0.0
    flag = 0.0 if res[1] in (0, 1, 2, 4, 5) else 1.0
    if nargout <= 1:
        if flag:
            interp.out(f"lsqr stopped at iteration {res[2]} without converging to the desired tolerance {tol:.2g}.\n")
        else:
            interp.out(f"lsqr converged at iteration {res[2]} to a solution with relative residual {relres:.2g}.\n")
        return x
    return (x, np.array([[flag]]), np.array([[relres]]), np.array([[float(res[2])]]))[:nargout]


# -- plotting ---------------------------------------------------------------------------------------------

@sbuiltin("spy")
def _spy(interp, args, nargout):
    m = as_csc(args[0])
    coo = m.tocoo()
    marker = "."
    extra: list = []
    for a in args[1:]:
        if is_text(a):
            marker = text(a)
        else:
            extra = [char("MarkerSize"), a]
    frame = interp.frame
    interp.call_named("plot", [(coo.col + 1.0).reshape(1, -1), (coo.row + 1.0).reshape(1, -1), char(marker), *extra],
                      0, frame)
    for name, cargs in (("axis", [char("ij")]), ("axis", [char("equal")]),
                        ("xlim", [np.array([[0.0, m.shape[1] + 1.0]])]),
                        ("ylim", [np.array([[0.0, m.shape[0] + 1.0]])]),
                        ("xlabel", [char(f"nz = {m.nnz}")])):
        try:
            interp.call_named(name, cargs, 0, frame)
        except MatlabError:
            pass
    return None
