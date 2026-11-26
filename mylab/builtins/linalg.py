"""Linear algebra."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..errors import MatlabError
from ..ops import left_divide
from ..values import Struct, simplify, to_int
from ._registry import builtin
from ._util import fnum, is_text, nargs, text


def _square(a: np.ndarray, name: str) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise MatlabError(f"Matrix must be square for {name}.")


def _matrix(v, name: str) -> np.ndarray:
    """A 2-D floating point matrix (single stays single)."""
    a = fnum(v)
    if a.ndim != 2:
        raise MatlabError(f"{name}: input must be a 2-D matrix.")
    return a


def _work(a: np.ndarray) -> np.ndarray:
    """The matrix in double precision (real or complex) for LAPACK."""
    return a.astype(complex if a.dtype.kind == "c" else float)


def _is_single(*arrays) -> bool:
    return any(isinstance(a, np.ndarray) and a.dtype in (np.float32, np.complex64) for a in arrays)


def _out(r, single: bool = False):
    """Finish a result: drop a zero imaginary part and restore single precision."""
    r = simplify(np.asarray(r))
    if single and r.dtype.kind in "fc":
        r = r.astype(np.complex64 if r.dtype.kind == "c" else np.float32)
    return r


def _split_flags(args: list, allowed: tuple, name: str) -> tuple[list, list]:
    """Separate text options (lower-cased, checked against ``allowed``) from the other arguments."""
    rest, flags = [], []
    for a in args:
        if is_text(a):
            s = text(a).lower()
            if s not in allowed:
                raise MatlabError(f"{name}: unknown option '{text(a)}'.")
            flags.append(s)
        else:
            rest.append(a)
    return rest, flags


def _tol_arg(v, name: str) -> float:
    t = np.real(fnum(v))
    if t.size != 1 or not t.flat[0] >= 0:
        raise MatlabError(f"{name}: the tolerance must be a nonnegative scalar.")
    return float(t.flat[0])


def _is_hermitian(a: np.ndarray) -> bool:
    return a.shape[0] == a.shape[1] and bool(np.array_equal(a, a.conj().T))


@builtin("inv")
def _inv(interp, args, nargout):
    nargs(args, 1, 1, "inv")
    a = fnum(args[0])
    _square(a, "inv")
    single = _is_single(a)
    w = _work(a)
    if w.size and (np.linalg.cond(w) > 1 / np.finfo(np.float32 if single else float).eps):
        interp.warn("Matrix is singular to working precision.", "MATLAB:singularMatrix")
        with np.errstate(all="ignore"):
            try:
                return _out(np.linalg.inv(w), single)
            except np.linalg.LinAlgError:
                return _out(np.full(a.shape, np.inf), single)
    return _out(np.linalg.inv(w), single)


@builtin("pinv")
def _pinv(interp, args, nargout):
    nargs(args, 1, 2, "pinv")
    a = _matrix(args[0], "pinv")
    single = _is_single(a)
    w = _work(a)
    if w.size == 0:
        return np.zeros((a.shape[1], a.shape[0]), dtype=a.dtype)
    if not np.all(np.isfinite(w)):
        raise MatlabError("pinv: input must not contain NaN or Inf.")
    u, s, vh = np.linalg.svd(w, full_matrices=False)
    if len(args) > 1:
        tol = _tol_arg(args[1], "pinv")
    else:
        top = s[0] if s.size else 0.0
        tol = max(a.shape) * np.spacing(np.float32(top) if single else top)
    keep = s > tol
    inv_s = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    return _out((vh.conj().T * inv_s) @ u.conj().T, single)


@builtin("det")
def _det(interp, args, nargout):
    nargs(args, 1, 1, "det")
    a = fnum(args[0])
    _square(a, "det")
    if a.size == 0:
        return 1.0
    return _out(np.asarray(np.linalg.det(_work(a))).reshape(1, 1), _is_single(a))


@builtin("rank")
def _rank(interp, args, nargout):
    nargs(args, 1, 2, "rank")
    a = _matrix(args[0], "rank")
    if a.size == 0:
        return 0.0
    s = np.linalg.svd(_work(a), compute_uv=False)
    if len(args) > 1:
        tol = _tol_arg(args[1], "rank")
    else:
        tol = max(a.shape) * np.spacing(np.float32(s[0]) if _is_single(a) else s[0])
    return float(np.sum(s > tol))


@builtin("trace")
def _trace(interp, args, nargout):
    nargs(args, 1, 1, "trace")
    a = fnum(args[0])
    _square(a, "trace")
    return _out(np.asarray(np.trace(a)).reshape(1, 1))


@builtin("norm")
def _norm(interp, args, nargout):
    nargs(args, 1, 2, "norm")
    a = fnum(args[0])
    if a.dtype in (np.float32, np.complex64):
        r = _norm(interp, [a.astype(complex if a.dtype.kind == "c" else float), *args[1:]], nargout)
        return np.array([[r]], dtype=np.float32)
    if a.ndim > 2:
        raise MatlabError("norm: input must be a vector or a 2-D matrix. Use vecnorm or pagenorm for N-D arrays.")
    p = args[1] if len(args) > 1 else None
    is_vec = a.shape[0] == 1 or a.shape[1] == 1
    if p is None:
        order = 2
    elif is_text(p):
        s = text(p).lower()
        order = {"fro": "fro", "inf": np.inf, "-inf": -np.inf}.get(s)
        if order is None:
            raise MatlabError(f"Unknown norm type '{text(p)}'. Use 1, 2, Inf or 'fro'.")
    else:
        pv = fnum(p)
        if pv.size != 1 or pv.dtype.kind == "c" or np.isnan(pv.flat[0]):
            raise MatlabError("norm: the norm type must be a real scalar, Inf, -Inf or 'fro'.")
        order = float(pv.flat[0])
    if a.size == 0:
        return 0.0
    if is_vec:
        v = np.abs(a.flatten())
        if order == "fro" or order == 2:
            return float(np.linalg.norm(v))
        if order == np.inf:
            return float(v.max())
        if order == -np.inf:
            return float(v.min())
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if order == 0:
                return float(np.float64(v.size) ** np.inf) if v.size > 1 else 1.0
            if order < 0 and np.any(v == 0):
                return 0.0
            scale = v.max()
            if scale == 0 or not np.isfinite(scale):
                return float(scale)
            return float(scale * np.sum((v / scale) ** order) ** (1.0 / order))
    if order not in (1, 2, np.inf, "fro"):
        raise MatlabError("norm: for a matrix the norm type must be 1, 2, Inf or 'fro'.")
    if np.isnan(a).any():
        return float("nan")
    if order == 2 and np.isinf(a).any():
        return float("inf")
    return float(np.linalg.norm(a, ord=order))


# -- eigenvalues ------------------------------------------------------------------------------

def _triangular_eigvecs(t: np.ndarray) -> np.ndarray:
    """Right eigenvectors of an upper triangular matrix by back substitution (unit 2-norm)."""
    n = t.shape[0]
    x = np.zeros((n, n), dtype=complex)
    small = np.finfo(float).eps * max(np.abs(t).max() if n else 0.0, np.finfo(float).tiny)
    for k in range(n):
        x[k, k] = 1.0
        for i in range(k - 1, -1, -1):
            d = t[i, i] - t[k, k]
            if abs(d) < small:
                d = small
            x[i, k] = -(t[i, i + 1:k + 1] @ x[i + 1:k + 1, k]) / d
        x[:, k] /= np.linalg.norm(x[:, k])
    return x


def _eig_nobalance(a: np.ndarray, want_left: bool):
    """Eigen-decomposition from the complex Schur form, with no balancing step."""
    t, z = sla.schur(a, output="complex")
    w = np.diag(t).copy()
    v = z @ _triangular_eigvecs(t)
    v /= np.linalg.norm(v, axis=0)
    wl = None
    if want_left:
        # left eigenvectors of T are the right eigenvectors of T^H (lower triangular): flip to reuse
        n = t.shape[0]
        flip = np.arange(n)[::-1]
        th = t.conj().T[np.ix_(flip, flip)]
        y = _triangular_eigvecs(th)[np.ix_(flip, flip)]
        wl = z @ y
        wl /= np.linalg.norm(wl, axis=0)
    return w, v, wl


def _ggev_scale(v: np.ndarray) -> np.ndarray:
    """Scale each column so that its largest component has |re| + |im| = 1 (the LAPACK convention)."""
    if v.size == 0:
        return v
    big = np.max(np.abs(v.real) + np.abs(v.imag), axis=0)
    big[big == 0] = 1.0
    return v / big


@builtin("eig")
def _eig(interp, args, nargout):
    nargs(args, 1, 5, "eig")
    if nargout > 3:
        raise MatlabError("eig: too many output arguments.")
    mats, flags = _split_flags(args, ("balance", "nobalance", "chol", "qz", "vector", "matrix"), "eig")
    if not mats or len(mats) > 2:
        raise MatlabError("eig: expected one matrix A, or two matrices A and B.")
    a = fnum(mats[0])
    _square(a, "eig")
    single = _is_single(a)
    a = _work(a)
    generalized = len(mats) == 2
    if generalized:
        b = fnum(mats[1])
        if b.shape != a.shape:
            raise MatlabError("eig: A and B must be square matrices of the same size.")
        single = single or _is_single(b)
        b = _work(b)
        if "balance" in flags or "nobalance" in flags:
            raise MatlabError("eig: the balance options do not apply to the generalized problem eig(A,B).")
    elif "chol" in flags or "qz" in flags:
        raise MatlabError("eig: the 'chol' and 'qz' options need two matrices: eig(A,B,algorithm).")
    if ("vector" in flags and "matrix" in flags) or ("balance" in flags and "nobalance" in flags) \
            or ("chol" in flags and "qz" in flags):
        raise MatlabError("eig: conflicting options.")
    if not np.all(np.isfinite(a)) or (generalized and not np.all(np.isfinite(b))):
        raise MatlabError("Input matrix contains NaN or Inf.", "MATLAB:eig:matrixWithNaNInf")
    want_vecs = nargout >= 2
    want_left = nargout >= 3
    v = wl = None
    if not generalized:
        if _is_hermitian(a):
            if want_vecs:
                w, v = np.linalg.eigh(a)
                wl = v
            else:
                w = np.linalg.eigvalsh(a)
        elif "nobalance" in flags:
            w, v, wl = _eig_nobalance(a, want_left)
        elif want_left:
            w, wl, v = sla.eig(a, left=True, right=True)
        elif want_vecs:
            w, v = sla.eig(a)
        else:
            w = sla.eigvals(a)
    else:
        use_chol = "qz" not in flags and _is_hermitian(a) and _is_hermitian(b)
        done = False
        if use_chol:
            try:
                if want_vecs:
                    w, v = sla.eigh(a, b)
                    wl = v
                else:
                    w = sla.eigvalsh(a, b)
                done = True
            except np.linalg.LinAlgError:  # B is not positive definite: fall back on QZ
                done = False
        if not done:
            with np.errstate(divide="ignore", invalid="ignore"):
                if want_left:
                    w, wl, v = sla.eig(a, b, left=True, right=True)
                elif want_vecs:
                    w, v = sla.eig(a, b)
                else:
                    w = sla.eigvals(a, b)
            if want_vecs:
                v = _ggev_scale(np.asarray(v, dtype=complex))
            if want_left:
                wl = _ggev_scale(np.asarray(wl, dtype=complex))
    w = _out(np.asarray(w), single)
    as_matrix = ("matrix" in flags) or (nargout >= 2 and "vector" not in flags)
    d = np.diag(w) if as_matrix else w.reshape(-1, 1)
    if nargout <= 1:
        return d
    # real eigenvectors for real eigenvalues of a real problem
    v = _out(v, single)
    if not want_left:
        return v, d
    return v, d, _out(wl, single)


# -- factorizations ---------------------------------------------------------------------------

@builtin("svd")
def _svd(interp, args, nargout):
    nargs(args, 1, 3, "svd")
    if nargout > 3:
        raise MatlabError("svd: too many output arguments.")
    a = _matrix(args[0], "svd")
    single = _is_single(a)
    w = _work(a)
    econ = False
    form = None
    for extra in args[1:]:
        if is_text(extra):
            s = text(extra).lower()
            if s == "econ":
                econ = True
            elif s in ("vector", "matrix"):
                form = s
            else:
                raise MatlabError(f"svd: unknown option '{text(extra)}'. Use 'econ', 0, 'vector' or 'matrix'.")
        else:
            z = fnum(extra)
            if z.size != 1 or z.flat[0] != 0:
                raise MatlabError("svd: the second argument must be 0 or 'econ'.")
            econ = econ or a.shape[0] > a.shape[1]  # svd(A, 0) only trims a tall matrix
    if not np.all(np.isfinite(w)):
        raise MatlabError("Input to SVD must not contain NaN or Inf.", "MATLAB:svd:matrixWithNaNInf")
    m, n = a.shape
    k = min(m, n)
    if nargout <= 1:
        s = np.linalg.svd(w, compute_uv=False) if w.size else np.zeros(0)
        s = _out(s, single)
        if form == "matrix":
            S = np.zeros((k, k) if econ else (m, n), dtype=s.dtype)
            S[np.arange(k), np.arange(k)] = s
            return S
        return s.reshape(-1, 1)
    if w.size == 0:
        u, s, vh = np.eye(m), np.zeros(0), np.eye(n)
        if econ:
            u, vh = u[:, :k], vh[:k, :]
    else:
        u, s, vh = np.linalg.svd(w, full_matrices=not econ)
    if form == "vector":
        S = s.reshape(-1, 1)
    else:
        S = np.zeros((u.shape[1], vh.shape[0]))
        S[np.arange(s.size), np.arange(s.size)] = s
    if nargout == 2:
        return _out(u, single), _out(S, single)
    return _out(u, single), _out(S, single), _out(vh.conj().T, single)


@builtin("qr")
def _qr(interp, args, nargout):
    nargs(args, 1, 3, "qr")
    if nargout > 3:
        raise MatlabError("qr: too many output arguments.")
    a = _matrix(args[0], "qr")
    single = _is_single(a)
    w = _work(a)
    econ = False
    perm_form = None
    for extra in args[1:]:
        if is_text(extra):
            s = text(extra).lower()
            if s == "econ":
                econ = True
            elif s in ("vector", "matrix"):
                perm_form = s
            else:
                raise MatlabError(f"qr: unknown option '{text(extra)}'. Use 0, 'econ', 'vector' or 'matrix'.")
        else:
            z = fnum(extra)
            if z.size != 1 or z.flat[0] != 0:
                raise MatlabError("qr: the second argument must be 0, 'econ', 'vector' or 'matrix'. "
                                  "Solving with a right-hand side, qr(A,B), is only available for sparse A.")
            econ = True
            perm_form = perm_form or "vector"  # qr(A, 0) returns the permutation as a vector
    mode = "economic" if econ else "full"
    if nargout <= 1:
        r = sla.qr(w, mode="r")[0] if w.size else np.zeros(a.shape)
        if econ:
            r = r[: min(a.shape), :]
        return _out(np.triu(r), single)
    if nargout == 2:
        q, r = sla.qr(w, mode=mode)
        return _out(q, single), _out(r, single)
    q, r, p = sla.qr(w, mode=mode, pivoting=True)
    if perm_form == "vector":
        pm = (p + 1).astype(float).reshape(1, -1)
    else:
        pm = np.eye(a.shape[1])[:, p]
    return _out(q, single), _out(r, single), pm


@builtin("lu")
def _lu(interp, args, nargout):
    nargs(args, 1, 2, "lu")
    if nargout > 3:
        raise MatlabError("lu: too many output arguments for a full matrix.")
    a = _matrix(args[0], "lu")
    single = _is_single(a)
    w = _work(a)
    vector = False
    if len(args) > 1:
        if not is_text(args[1]) or text(args[1]).lower() not in ("vector", "matrix"):
            raise MatlabError("lu: the second argument must be 'vector' or 'matrix'.")
        vector = text(args[1]).lower() == "vector"
    m, n = w.shape
    k = min(m, n)
    if w.size == 0:
        l, u, pidx = np.zeros((m, k)), np.zeros((k, n)), np.arange(m)
    else:
        pidx, l, u = sla.lu(w, p_indices=True)  # A = L[pidx, :] @ U
    if nargout <= 1:
        # one output: both factors packed into one matrix (L below the diagonal, U on and above it)
        packed = np.zeros((m, n), dtype=u.dtype)
        packed[:k, :] = u
        packed[:, :k] += np.tril(l, -1)
        return _out(packed, single)
    if nargout == 2:
        return _out(l[pidx, :], single), _out(u, single)
    perm = np.argsort(pidx)  # L*U = A(perm, :)
    if vector:
        return _out(l, single), _out(u, single), (perm + 1).astype(float).reshape(1, -1)
    return _out(l, single), _out(u, single), np.eye(m)[perm, :]


@builtin("chol")
def _chol(interp, args, nargout):
    nargs(args, 1, 3, "chol")
    if nargout > 2:
        raise MatlabError("chol: too many output arguments for a full matrix.")
    a = fnum(args[0])
    _square(a, "chol")
    single = _is_single(a)
    w = _work(a)
    lower = False
    for extra in args[1:]:
        s = text(extra).lower() if is_text(extra) else None
        if s == "lower":
            lower = True
        elif s == "upper":
            lower = False
        elif s in ("vector", "matrix"):
            raise MatlabError(f"chol: the '{s}' option is only available for sparse matrices.")
        else:
            raise MatlabError("chol: the triangle option must be 'upper' or 'lower'.")
    n = w.shape[0]
    if n == 0:
        return (np.zeros((0, 0)), 0.0) if nargout > 1 else np.zeros((0, 0))
    # Only one triangle of A is read: the upper one by default, the lower one with 'lower'.
    tri = np.tril(w) if lower else np.triu(w)
    herm = tri + tri.conj().T
    herm[np.arange(n), np.arange(n)] = np.real(np.diag(w))
    flag = 0
    if np.all(np.isfinite(herm)):
        try:
            f = sla.cholesky(herm, lower=lower)
        except np.linalg.LinAlgError:
            f = None
    else:
        f = None
    if f is None:
        # find the first leading block that is not positive definite
        for q in range(1, n + 1):
            lead = herm[:q, :q]
            ok = np.all(np.isfinite(lead))
            if ok:
                try:
                    sla.cholesky(lead, lower=lower)
                except np.linalg.LinAlgError:
                    ok = False
            if not ok:
                flag = q
                break
        if nargout < 2:
            raise MatlabError("Matrix must be positive definite.", "MATLAB:posdef")
        f = sla.cholesky(herm[:flag - 1, :flag - 1], lower=lower) if flag > 1 else np.zeros((0, 0))
    r = _out(f, single)
    return (r, float(flag)) if nargout > 1 else r


# -- matrix functions ---------------------------------------------------------------------------

@builtin("expm")
def _expm(interp, args, nargout):
    nargs(args, 1, 1, "expm")
    a = fnum(args[0])
    _square(a, "expm")
    if a.size == 0:
        return np.zeros((0, 0))
    return _out(sla.expm(_work(a)), _is_single(a))


@builtin("logm")
def _logm(interp, args, nargout):
    nargs(args, 1, 1, "logm")
    if nargout > 2:
        raise MatlabError("logm: too many output arguments.")
    a = fnum(args[0])
    _square(a, "logm")
    w = _work(a)
    if w.size == 0:
        return (np.zeros((0, 0)), 0.0) if nargout > 1 else np.zeros((0, 0))
    ev = np.linalg.eigvals(w)
    flag = 0.0
    if np.any(ev == 0):
        interp.warn("Principal matrix logarithm is not defined for A with nonpositive real "
                    "eigenvalues. A non-principal matrix logarithm is returned.", "MATLAB:logm:nonPosRealEig")
        flag = 1.0
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = np.asarray(sla.logm(w))
    if not np.all(np.isfinite(r)):
        flag = 1.0
    r = _out(r, _is_single(a))
    return (r, flag) if nargout > 1 else r


@builtin("sqrtm")
def _sqrtm(interp, args, nargout):
    nargs(args, 1, 1, "sqrtm")
    if nargout > 3:
        raise MatlabError("sqrtm: too many output arguments.")
    a = fnum(args[0])
    _square(a, "sqrtm")
    w = _work(a)
    n = w.shape[0]
    if n == 0:
        outs = (np.zeros((0, 0)), 0.0, 0.0)
        return outs[:nargout] if nargout > 1 else outs[0]
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        x = np.asarray(sla.sqrtm(w))
    singular = not np.all(np.isfinite(x)) or np.any(np.abs(np.linalg.eigvals(w)) == 0)
    if nargout <= 1:
        if singular:
            interp.warn("Matrix is singular and may not have a square root.", "MATLAB:sqrtm:SingularMatrix")
        return _out(x, _is_single(a))
    xs = _out(x, _is_single(a))
    anorm = np.linalg.norm(w, "fro")
    if nargout == 2:
        res = np.linalg.norm(w - x @ x, 1) / np.linalg.norm(w, 1) if np.linalg.norm(w, 1) else 0.0
        return xs, float(res)
    # stability factor and the Frobenius-norm condition number of the square root at A
    alpha = np.linalg.norm(x, "fro") ** 2 / anorm if anorm else np.inf
    try:
        k = np.kron(np.eye(n), x) + np.kron(x.T, np.eye(n))
        smin = np.linalg.svd(k, compute_uv=False)[-1]
        condx = (anorm / np.linalg.norm(x, "fro")) / smin if smin > 0 else np.inf
    except np.linalg.LinAlgError:
        condx = np.inf
    return xs, float(alpha), float(condx)


# -- subspaces and solving ---------------------------------------------------------------------

def _rational_null(a: np.ndarray) -> np.ndarray:
    """Null-space basis read off the reduced row echelon form (not orthonormal)."""
    from .linalg_more import rref_array
    r, pivots = rref_array(a, None)
    n = a.shape[1]
    free = [j for j in range(n) if j not in pivots]
    z = np.zeros((n, len(free)), dtype=r.dtype)
    for k, j in enumerate(free):
        z[j, k] = 1.0
        for row, p in enumerate(pivots):
            z[p, k] = -r[row, j]
    return z


@builtin("null")
def _null(interp, args, nargout):
    nargs(args, 1, 2, "null")
    a = _matrix(args[0], "null")
    single = _is_single(a)
    w = _work(a)
    m, n = w.shape
    tol = None
    if len(args) > 1:
        if is_text(args[1]):
            if text(args[1]).lower() not in ("r", "rational"):
                raise MatlabError("null: the second argument must be a tolerance or 'rational'.")
            return _out(_rational_null(w), single)
        tol = _tol_arg(args[1], "null")
    if n == 0:
        return np.zeros((0, 0))
    if m == 0:
        return np.eye(n)
    u, s, vh = np.linalg.svd(w, full_matrices=True)
    if tol is None:
        top = s[0] if s.size else 0.0
        tol = max(m, n) * np.spacing(np.float32(top) if single else top)
    r = int(np.sum(s > tol))
    return _out(vh[r:, :].conj().T, single)


@builtin("orth")
def _orth(interp, args, nargout):
    nargs(args, 1, 2, "orth")
    a = _matrix(args[0], "orth")
    single = _is_single(a)
    w = _work(a)
    m, n = w.shape
    if w.size == 0:
        return np.zeros((m, 0))
    u, s, _ = np.linalg.svd(w, full_matrices=False)
    if len(args) > 1:
        tol = _tol_arg(args[1], "orth")
    else:
        top = s[0] if s.size else 0.0
        tol = max(m, n) * np.spacing(np.float32(top) if single else top)
    r = int(np.sum(s > tol))
    return _out(u[:, :r], single)


_LINSOLVE_FIELDS = ("LT", "UT", "UHESS", "SYM", "POSDEF", "RECT", "TRANSA")


@builtin("linsolve")
def _linsolve(interp, args, nargout):
    nargs(args, 2, 3, "linsolve")
    if nargout > 2:
        raise MatlabError("linsolve: too many output arguments.")
    a = _matrix(args[0], "linsolve")
    b = _matrix(args[1], "linsolve")
    single = _is_single(a, b)
    a, b = _work(a), _work(b)
    opts = dict.fromkeys(_LINSOLVE_FIELDS, False)
    if len(args) > 2:
        s = args[2]
        if not isinstance(s, Struct):
            raise MatlabError("linsolve: the options must be a structure with fields such as LT, UT, "
                              "SYM, POSDEF, RECT and TRANSA.")
        for f in s.fields:
            if f not in opts:
                raise MatlabError(f"linsolve: unknown option field '{f}'.")
            v = s.get_field(0, f)
            opts[f] = bool(np.asarray(v).size and np.all(np.asarray(v) != 0))
    if opts["LT"] and opts["UT"]:
        raise MatlabError("linsolve: LT and UT cannot both be true.")
    if (opts["LT"] or opts["UT"]) and (opts["SYM"] or opts["POSDEF"] or opts["UHESS"]):
        raise MatlabError("linsolve: this combination of options is not supported.")
    if opts["POSDEF"] and not opts["SYM"]:
        opts["SYM"] = True
    trans = opts["TRANSA"]
    m, n = a.shape
    op = a.conj().T if trans else a
    if b.shape[0] != op.shape[0]:
        raise MatlabError("Matrix dimensions must agree.")
    rc = None
    if m != n:
        if opts["SYM"] or opts["UHESS"]:
            raise MatlabError("linsolve: SYM, POSDEF and UHESS need a square matrix.")
        x = left_divide(op, b)  # least squares by QR with column pivoting
        if nargout > 1:
            rc = float(np.linalg.matrix_rank(a))
    elif opts["LT"] or opts["UT"]:
        if n and np.any(np.diag(a) == 0):
            interp.warn("Matrix is singular to working precision.", "MATLAB:singularMatrix")
        with np.errstate(all="ignore"):
            x = sla.solve_triangular(a, b, lower=opts["LT"], trans=2 if trans else 0, check_finite=False) \
                if n else np.zeros((0, b.shape[1]))
        if nargout > 1:
            trcon = sla.lapack.get_lapack_funcs("trcon", (a,))
            rc = float(trcon(a, norm="1", uplo="L" if opts["LT"] else "U")[0]) if n else np.inf
    else:
        if opts["POSDEF"]:
            try:
                c = sla.cho_factor(op, check_finite=False)
            except np.linalg.LinAlgError:
                raise MatlabError("linsolve: the matrix must be positive definite when POSDEF is true.")
            x = sla.cho_solve(c, b, check_finite=False)
        elif opts["SYM"]:
            x = sla.solve(op, b, assume_a="her" if op.dtype.kind == "c" else "sym", check_finite=False)
        else:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x = sla.solve(op, b, check_finite=False)
            except np.linalg.LinAlgError:
                interp.warn("Matrix is singular to working precision.", "MATLAB:singularMatrix")
                x = np.full((n, b.shape[1]), np.inf)
        if nargout > 1:
            from ._registry import BUILTINS
            rc = float(np.asarray(BUILTINS["rcond"](interp, [op], 1)).flat[0])
    x = _out(x, single)
    return (x, rc) if nargout > 1 else x


@builtin("cond")
def _cond(interp, args, nargout):
    nargs(args, 1, 2, "cond")
    a = _matrix(args[0], "cond")
    single = _is_single(a)
    w = _work(a)
    p = 2
    if len(args) > 1:
        if is_text(args[1]):
            s = text(args[1]).lower()
            if s == "fro":
                p = "fro"
            elif s == "inf":
                p = np.inf
            else:
                raise MatlabError(f"cond: unknown norm type '{text(args[1])}'. Use 1, 2, Inf or 'fro'.")
        else:
            pv = fnum(args[1])
            if pv.size != 1 or pv.flat[0] not in (1, 2, np.inf):
                raise MatlabError("cond: the norm type must be 1, 2, Inf or 'fro'.")
            p = pv.flat[0]
            p = np.inf if np.isinf(p) else int(p)
    if w.size == 0:
        return 0.0
    if not np.all(np.isfinite(w)):
        if np.isnan(w).any():
            return _out(np.array([[np.nan]]), single)
    if p == 2:
        s = np.linalg.svd(w, compute_uv=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.inf if s[-1] == 0 else s[0] / s[-1]
        return _out(np.array([[c]]), single)
    if w.shape[0] != w.shape[1]:
        raise MatlabError("cond: the matrix must be square for the 1, Inf and Frobenius norms.")
    try:
        with np.errstate(all="ignore"):
            inv = np.linalg.inv(w)
        c = np.linalg.norm(w, p) * np.linalg.norm(inv, p)
        if not np.isfinite(c):
            c = np.inf
    except np.linalg.LinAlgError:
        c = np.inf
    return _out(np.array([[c]]), single)


# -- special matrices ---------------------------------------------------------------------------

@builtin("vander")
def _vander(interp, args, nargout):
    nargs(args, 1, 1, "vander")
    v = fnum(args[0])
    if v.ndim != 2 or (v.size and 1 not in v.shape):
        raise MatlabError("vander: input must be a vector.")
    return np.vander(v.flatten())


@builtin("hilb")
def _hilb(interp, args, nargout):
    nargs(args, 1, 2, "hilb")
    n = to_int(args[0], "n")
    if n < 0:
        n = 0
    h = sla.hilbert(n) if n else np.zeros((0, 0))
    if len(args) > 1:
        cls = text(args[1])
        if cls == "single":
            return h.astype(np.float32)
        if cls != "double":
            raise MatlabError("hilb: the class must be 'double' or 'single'.")
    return h


@builtin("toeplitz")
def _toeplitz(interp, args, nargout):
    nargs(args, 1, 2, "toeplitz")
    c = fnum(args[0])
    if c.ndim != 2 or (c.size and 1 not in c.shape):
        raise MatlabError("toeplitz: inputs must be vectors.")
    c = c.flatten()
    if len(args) > 1:
        r = fnum(args[1])
        if r.ndim != 2 or (r.size and 1 not in r.shape):
            raise MatlabError("toeplitz: inputs must be vectors.")
        r = r.flatten()
        if c.size and r.size and c[0] != r[0]:
            interp.warn("First element of input column does not match first element of input row. "
                        "Column wins diagonal conflict.", "MATLAB:toeplitz:DiagonalConflict")
        # first column c, first row r (not conjugated); the column gives the diagonal
        n, m = c.size, r.size
        if n == 0 or m == 0:
            return np.zeros((n, m))
        i, j = np.indices((n, m))
        both = np.concatenate([r[:0:-1], c])
        return both[(i - j) + (m - 1)]
    # one argument: the first row is r and the matrix is Hermitian off the diagonal
    n = c.size
    i, j = np.indices((n, n))
    d = j - i
    t = np.where(d >= 0, c[np.abs(d)], np.conj(c[np.abs(d)])) if n else np.zeros((0, 0))
    return simplify(t)


@builtin("pascal")
def _pascal(interp, args, nargout):
    from math import comb
    nargs(args, 1, 3, "pascal")
    n = to_int(args[0], "n")
    if n < 0:
        raise MatlabError("pascal: n must be a nonnegative integer.")
    cls = None
    rest = list(args[1:])
    if rest and is_text(rest[-1]):
        cls = text(rest.pop())
        if cls not in ("double", "single"):
            raise MatlabError("pascal: the class must be 'double' or 'single'.")
    kind = to_int(rest[0], "type") if rest else 0
    if kind == 0:
        # P(i, j) = binomial(i + j - 2, j - 1): symmetric positive definite
        p = np.array([[float(comb(i + j, j)) for j in range(n)] for i in range(n)]).reshape(n, n)
    elif kind in (1, 2):
        # lower-triangular factor with alternating column signs; it is its own inverse
        p = np.array([[float((-1) ** j * comb(i, j)) if j <= i else 0.0 for j in range(n)]
                      for i in range(n)]).reshape(n, n)
        if kind == 2:
            # a rotated copy of the type-1 matrix: a cube root of the identity
            p = np.rot90(p, 3)
            if n % 2 == 0:
                p = -p
    else:
        raise MatlabError("pascal: type must be 0, 1 or 2.")
    return p.astype(np.float32) if cls == "single" else p
