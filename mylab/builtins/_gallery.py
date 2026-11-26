"""Test-matrix families for ``gallery`` (imported on first use by linalg_more).

Every family is written from its published mathematical definition.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import simplify
from ._util import fnum, is_text, text

_EPS = float(np.finfo(float).eps)


def _int(v: Any, what: str) -> int:
    a = np.real(fnum(v))
    if a.size != 1 or a.flat[0] != np.round(a.flat[0]):
        raise MatlabError(f"gallery: {what} must be an integer.")
    return int(a.flat[0])


def _num(v: Any, what: str):
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"gallery: {what} must be a scalar.")
    x = a.flat[0]
    return complex(x) if a.dtype.kind == "c" and x.imag != 0 else float(np.real(x))


def _opt(args: list, k: int, default: Any = None) -> Any:
    if len(args) > k and not (isinstance(args[k], np.ndarray) and args[k].size == 0):
        return args[k]
    return default


def _vec_or_n(v: Any, what: str) -> np.ndarray:
    """A vector argument; a scalar n stands for 1:n."""
    a = fnum(v)
    if a.size == 1:
        n = _int(v, what)
        return np.arange(1, n + 1, dtype=float)
    return a.flatten()


def _sparse(m):
    from .sparse import SparseMatrix
    return SparseMatrix(m)


def _rng(interp):
    """Uniform and normal generators that follow the interpreter's random stream."""
    from ._registry import BUILTINS

    def rand(*shape):
        return np.asarray(BUILTINS["rand"](interp, [np.array([[float(s)]]) for s in shape], 1), dtype=float)

    def randn(*shape):
        return np.asarray(BUILTINS["randn"](interp, [np.array([[float(s)]]) for s in shape], 1), dtype=float)

    return rand, randn


def _ij(n: int, m: int | None = None):
    i, j = np.indices((n, n if m is None else m))
    return i + 1.0, j + 1.0


# -- deterministic families ---------------------------------------------------------------------

def minij(interp, args):
    n = _int(args[0], "n")
    i, j = _ij(n)
    return np.minimum(i, j)


def tridiag(interp, args):
    import scipy.sparse as sp
    if len(args) == 1:
        n = _int(args[0], "n")
        c, d, e = -np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)
    elif len(args) == 3:
        c, d, e = (fnum(a).flatten() for a in args)
        n = d.size
    elif len(args) == 4:
        n = _int(args[0], "n")
        c, d, e = (fnum(a).flatten() for a in args[1:])
        if c.size == 1 and d.size == 1 and e.size == 1:
            c, d, e = np.full(n - 1, c[0]), np.full(n, d[0]), np.full(n - 1, e[0])
    else:
        raise MatlabError("gallery('tridiag', ...): expected n, or c, d, e, or n, c, d, e.")
    if d.size != n or c.size != max(n - 1, 0) or e.size != max(n - 1, 0):
        raise MatlabError("gallery('tridiag'): the sub- and superdiagonal must be one element shorter than "
                          "the diagonal.")
    return _sparse(sp.diags([c, d, e], [-1, 0, 1], shape=(n, n), format="csc"))


def poisson(interp, args):
    import scipy.sparse as sp
    n = _int(args[0], "n")
    t = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], shape=(n, n))
    eye = sp.identity(n)
    return _sparse(sp.kron(eye, t) + sp.kron(t, eye))


def lehmer(interp, args):
    n = _int(args[0], "n")
    i, j = _ij(n)
    return np.minimum(i, j) / np.maximum(i, j)


def triw(interp, args):
    nv = fnum(args[0]).flatten()
    m, n = (int(nv[0]), int(nv[-1]))
    alpha = _num(_opt(args, 1, np.array([[-1.0]])), "alpha")
    k = _int(_opt(args, 2, np.array([[n - 1.0]])), "k")
    t = np.tril(np.eye(m, n) + alpha * np.triu(np.ones((m, n)), 1), k)
    return t


def moler(interp, args):
    n = _int(args[0], "n")
    alpha = _num(_opt(args, 1, np.array([[-1.0]])), "alpha")
    u = np.eye(n) + alpha * np.triu(np.ones((n, n)), 1)
    return simplify(u.conj().T @ u)


def frank(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[0.0]])), "k")
    i, j = _ij(n)
    f = np.where(j >= i - 1, n + 1 - np.maximum(i, j), 0.0)
    if k == 1:
        f = f[::-1, ::-1].T  # reflect about the anti-diagonal
    elif k != 0:
        raise MatlabError("gallery('frank'): k must be 0 or 1.")
    return f


def kahan(interp, args):
    nv = fnum(args[0]).flatten()
    m, n = int(nv[0]), int(nv[-1])
    theta = _num(_opt(args, 1, np.array([[1.2]])), "theta")
    pert = _num(_opt(args, 2, np.array([[1e3]])), "pert")
    s, c = np.sin(theta), np.cos(theta)
    u = np.eye(n) - c * np.triu(np.ones((n, n)), 1)
    u = np.diag(s ** np.arange(n)) @ u + pert * _EPS * np.diag(np.arange(n, 0, -1.0))
    if m > n:
        u = np.vstack([u, np.zeros((m - n, n))])
    elif m < n:
        u = u[:m, :]
    return u


def circul(interp, args):
    v = _vec_or_n(args[0], "v")
    n = v.size
    i, j = np.indices((n, n))
    return v[(j - i) % n]


def jordbloc(interp, args):
    n = _int(args[0], "n")
    lam = _num(_opt(args, 1, np.array([[1.0]])), "lambda")
    return lam * np.eye(n) + np.diag(np.ones(n - 1), 1)


def clement(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[0.0]])), "k")
    x = np.arange(n - 1, 0, -1.0)  # subdiagonal
    z = np.arange(1.0, n)          # superdiagonal
    if k == 0:
        return np.diag(x, -1) + np.diag(z, 1)
    if k == 1:
        y = np.sqrt(x * z)
        return np.diag(y, 1) + np.diag(y, -1)
    raise MatlabError("gallery('clement'): k must be 0 or 1.")


def fiedler(interp, args):
    c = _vec_or_n(args[0], "c")
    return np.abs(c.reshape(-1, 1) - c.reshape(1, -1))


def pei(interp, args):
    n = _int(args[0], "n")
    alpha = _num(_opt(args, 1, np.array([[1.0]])), "alpha")
    return alpha * np.eye(n) + np.ones((n, n))


def ris(interp, args):
    n = _int(args[0], "n")
    i, j = _ij(n)
    return 0.5 / (n - i - j + 1.5)


def cauchy(interp, args):
    x = _vec_or_n(args[0], "x")
    y = _vec_or_n(args[1], "y") if len(args) > 1 else x
    if x.size != y.size:
        raise MatlabError("gallery('cauchy'): x and y must have the same number of elements.")
    return 1.0 / (x.reshape(-1, 1) + y.reshape(1, -1))


def parter(interp, args):
    n = _int(args[0], "n")
    i, j = _ij(n)
    return 1.0 / (i - j + 0.5)


def lotkin(interp, args):
    n = _int(args[0], "n")
    i, j = _ij(n)
    a = 1.0 / (i + j - 1)
    if n:
        a[0, :] = 1.0
    return a


def toeppd(interp, args):
    n = _int(args[0], "n")
    m = _int(_opt(args, 1, np.array([[float(n)]])), "m")
    rand, _ = _rng(interp)
    w = fnum(_opt(args, 2, rand(m, 1))).flatten()
    theta = fnum(_opt(args, 3, rand(m, 1))).flatten()
    if w.size != m or theta.size != m:
        raise MatlabError("gallery('toeppd'): w and theta must have m elements.")
    i, j = np.indices((n, n))
    t = np.zeros((n, n))
    for k in range(m):
        t += w[k] * np.cos(2 * np.pi * theta[k] * (i - j))
    return t


def toeppen(interp, args):
    import scipy.sparse as sp
    n = _int(args[0], "n")
    vals = [_num(_opt(args, k + 1, np.array([[d]])), "coefficient") for k, d in enumerate((1.0, -10.0, 0.0, 10.0, 1.0))]
    diags = [np.full(n - abs(o), v) for v, o in zip(vals, (-2, -1, 0, 1, 2))]
    return _sparse(sp.diags(diags, [-2, -1, 0, 1, 2], shape=(n, n), format="csc"))


def house(interp, args, nargout):
    x = fnum(args[0])
    if x.ndim != 2 or x.shape[1] != 1:
        raise MatlabError("gallery('house'): x must be a column vector.")
    k = _int(_opt(args, 1, np.array([[0.0]])), "k")
    x = x.astype(complex if x.dtype.kind == "c" else float)
    v = x.copy()
    nrm = np.linalg.norm(x)
    if nrm == 0:
        s = 0.0
        beta = 1.0
    else:
        x1 = x[0, 0]
        sgn = x1 / abs(x1) if x1 != 0 else 1.0
        if k == 0:
            s = -sgn * nrm
        elif k == 1:
            s = sgn * nrm
        elif k == 2:
            if x.dtype.kind == "c":
                raise MatlabError("gallery('house'): k = 2 needs a real vector.")
            s = nrm
        else:
            raise MatlabError("gallery('house'): k must be 0, 1 or 2.")
        v[0, 0] = x1 - s
        beta = 0.0 if v[0, 0] == 0 else -1.0 / (np.conj(s) * v[0, 0])
    outs = (simplify(v), simplify(np.array([[beta]])), simplify(np.array([[s]])))
    return outs[:max(nargout, 1)] if nargout > 1 else outs[0]


def orthog(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[1.0]])), "k")
    i, j = _ij(n)
    if k == 1:
        return np.sqrt(2.0 / (n + 1)) * np.sin(i * j * np.pi / (n + 1))
    if k == 2:
        return 2.0 / np.sqrt(2 * n + 1) * np.sin(2 * i * j * np.pi / (2 * n + 1))
    if k == 3:
        return np.exp(2j * np.pi * (i - 1) * (j - 1) / n) / np.sqrt(n)
    if k == 4:
        # Helmert matrix: first row constant, row r has r-1 equal entries followed by a negative one
        q = np.tril(np.ones((n, n)))
        q[0, 1:] = 1.0
        for r in range(1, n):
            q[r, r] = -float(r)
        return q / np.sqrt(np.sum(q * q, axis=1, keepdims=True))
    if k == 5:
        ang = 2 * np.pi * (i - 1) * (j - 1) / n
        return (np.sin(ang) + np.cos(ang)) / np.sqrt(n)
    if k == 6:
        return np.sqrt(2.0 / n) * np.cos((i - 0.5) * (j - 0.5) * np.pi / n)
    if k == -1:
        return np.cos((i - 1) * (j - 1) * np.pi / (n - 1))
    if k == -2:
        return np.cos((i - 1) * (j - 0.5) * np.pi / n)
    raise MatlabError("gallery('orthog'): k must be one of 1 to 6, -1 or -2.")


def wilk(interp, args, nargout):
    n = _int(args[0], "n")
    if n == 3:
        a = np.array([[1e-10, 0.9, -0.4], [0.0, 0.9, -0.4], [0.0, 0.0, 1e-10]])
        b = np.zeros((3, 1))
    elif n == 4:
        a = np.array([[0.9143e-4, 0.0, 0.0, 0.0],
                      [0.8762, 0.7156e-4, 0.0, 0.0],
                      [0.7943, 0.8143, 0.9504e-4, 0.0],
                      [0.8017, 0.6123, 0.7165, 0.7123e-4]])
        b = np.array([[0.6524], [0.3127], [0.4186], [0.7853]])
    elif n == 5:
        i, j = _ij(6)
        h = 1.0 / (i + j - 1)
        a = h[0:5, 1:6] * 1.8144
        b = np.zeros((0, 0))
    elif n == 21:
        d = np.abs(np.arange(-10.0, 11.0))
        a = np.diag(d) + np.diag(np.ones(20), 1) + np.diag(np.ones(20), -1)
        b = np.zeros((0, 0))
    else:
        raise MatlabError("gallery('wilk'): n must be 3, 4, 5 or 21.")
    return (a, b) if nargout > 1 else a


def invhess(interp, args):
    x = _vec_or_n(args[0], "x")
    n = x.size
    y = fnum(args[1]).flatten() if len(args) > 1 else -x[: n - 1]
    if y.size != n - 1:
        raise MatlabError("gallery('invhess'): y must have one element fewer than x.")
    i, j = np.indices((n, n))
    return np.where(i >= j, x[j], y[np.minimum(i, n - 2)] if n > 1 else 0.0)


def kms(interp, args):
    n = _int(args[0], "n")
    rho = _num(_opt(args, 1, np.array([[0.5]])), "rho")
    i, j = np.indices((n, n))
    a = np.asarray(rho, dtype=complex if isinstance(rho, complex) else float) ** np.abs(i - j)
    if isinstance(rho, complex):
        a = np.triu(a).conj().T + np.triu(a, 1)
    return a


def grcar(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[3.0]])), "k")
    return np.tril(np.triu(np.ones((n, n))), k) - np.diag(np.ones(n - 1), -1)


def forsythe(interp, args):
    n = _int(args[0], "n")
    alpha = _num(_opt(args, 1, np.array([[np.sqrt(_EPS)]])), "alpha")
    lam = _num(_opt(args, 2, np.array([[0.0]])), "lambda")
    a = lam * np.eye(n) + np.diag(np.ones(n - 1), 1)
    if n:
        a = a.astype(complex) if isinstance(alpha, complex) or isinstance(lam, complex) else a
        a[n - 1, 0] = alpha
    return a


def gcdmat(interp, args):
    n = _int(args[0], "n")
    i, j = np.indices((n, n))
    return np.gcd(i + 1, j + 1).astype(float)


def hanowa(interp, args):
    n = _int(args[0], "n")
    d = _num(_opt(args, 1, np.array([[-1.0]])), "d")
    if n % 2:
        raise MatlabError("gallery('hanowa'): n must be even.")
    m = n // 2
    k = np.diag(np.arange(1.0, m + 1))
    return np.block([[d * np.eye(m), -k], [k, d * np.eye(m)]])


def lauchli(interp, args):
    n = _int(args[0], "n")
    mu = _num(_opt(args, 1, np.array([[np.sqrt(_EPS)]])), "mu")
    return np.vstack([np.ones((1, n)), mu * np.eye(n)])


def lesp(interp, args):
    n = _int(args[0], "n")
    x = np.arange(2.0, n + 1)
    return np.diag(1.0 / x, -1) + np.diag(-(2 * np.arange(1.0, n + 1) + 3)) + np.diag(x, 1)


def redheff(interp, args):
    n = _int(args[0], "n")
    i, j = np.indices((n, n))
    return ((np.mod(j + 1, i + 1) == 0) | (j == 0)).astype(float)


def riemann(interp, args):
    n = _int(args[0], "n")
    i, j = np.indices((n, n))
    i, j = i + 2, j + 2
    return np.where(np.mod(j, i) == 0, i - 1.0, -1.0)


def smoke(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[0.0]])), "k")
    w = np.exp(2j * np.pi / n)
    a = np.diag(w ** np.arange(1, n + 1)) + np.diag(np.ones(n - 1), 1)
    if k == 0:
        a[n - 1, 0] = 1.0
    elif k != 1:
        raise MatlabError("gallery('smoke'): k must be 0 or 1.")
    return a


def chow(interp, args):
    n = _int(args[0], "n")
    alpha = _num(_opt(args, 1, np.array([[1.0]])), "alpha")
    delta = _num(_opt(args, 2, np.array([[0.0]])), "delta")
    i, j = np.indices((n, n))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a = np.where(j <= i + 1, np.asarray(alpha, dtype=float) ** np.maximum(i - j + 1, 0), 0.0)
    return a + delta * np.eye(n)


def ipjfact(interp, args, nargout):
    from fractions import Fraction
    from math import factorial
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[0.0]])), "k")
    if k not in (0, 1):
        raise MatlabError("gallery('ipjfact'): k must be 0 or 1.")
    exact = [[Fraction(factorial(i + j + 2)) if k == 0 else Fraction(1, factorial(i + j + 2))
              for j in range(n)] for i in range(n)]
    a = np.array([[float(x) for x in row] for row in exact]).reshape(n, n)
    if nargout <= 1:
        return a
    # exact determinant by fraction-preserving elimination
    m = [row[:] for row in exact]
    det = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if m[r][c] != 0), None)
        if piv is None:
            det = Fraction(0)
            break
        if piv != c:
            m[c], m[piv] = m[piv], m[c]
            det = -det
        det *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for cc in range(c, n):
                m[r][cc] -= f * m[c][cc]
    return a, float(det)


def prolate(interp, args):
    n = _int(args[0], "n")
    w = _num(_opt(args, 1, np.array([[0.25]])), "w")
    k = np.arange(1.0, n)
    a = np.concatenate([[2 * w], np.sin(2 * np.pi * w * k) / (np.pi * k)])
    i, j = np.indices((n, n))
    return a[np.abs(i - j)]


def gearmat(interp, args):
    n = _int(args[0], "n")
    i = _int(_opt(args, 1, np.array([[float(n)]])), "i")
    j = _int(_opt(args, 2, np.array([[-float(n)]])), "j")
    if not (0 < abs(i) <= n and 0 < abs(j) <= n):
        raise MatlabError("gallery('gearmat'): |i| and |j| must be between 1 and n.")
    a = np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 1), 1)
    a[0, abs(i) - 1] = np.sign(i)
    a[n - 1, n - abs(j)] = np.sign(j)
    return a


def dramadah(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[1.0]])), "k")
    i, j = np.indices((n, n))
    if k == 1:
        # Toeplitz: first column 1 0 0 1 1 0 0 1 1 ..., first row 1 1 0 1 0 0 ...
        c = np.ones(n)
        for start in range(1, n, 4):
            c[start:start + 2] = 0.0
        r = np.zeros(n)
        r[:min(4, n)] = [1.0, 1.0, 0.0, 1.0][:min(4, n)]
        both = np.concatenate([r[:0:-1], c])
        return both[(i - j) + (n - 1)]
    if k == 2:
        # upper triangular Toeplitz with first row 1 1 0 1 0 1 0 ...
        r = np.zeros(n)
        r[0] = 1.0
        r[1::2] = 1.0
        return np.where(j >= i, r[np.abs(j - i)], 0.0)
    if k == 3:
        # lower Hessenberg (0,1) matrix
        r = np.zeros(n)
        r[0::2] = 1
        a = np.where(i >= j, r[np.abs(i - j)], 0.0)
        return a + np.diag(np.ones(n - 1), 1)
    raise MatlabError("gallery('dramadah'): k must be 1, 2 or 3.")


def neumann(interp, args):
    import scipy.sparse as sp
    n = _int(args[0], "n")
    m = int(round(np.sqrt(n)))
    if m * m != n:
        raise MatlabError("gallery('neumann'): n must be a perfect square.")
    t = sp.lil_matrix(sp.diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1], shape=(m, m)))
    if m > 1:
        t[0, 1] = -2
        t[m - 1, m - 2] = -2
    t = sp.csc_matrix(t)
    eye = sp.identity(m)
    return _sparse(sp.kron(t, eye) + sp.kron(eye, t))


def binomial(interp, args):
    from math import comb
    n = _int(args[0], "n")
    # A(i,j) = coefficient of x^(j-1) in (1+x)^(n-i) (1-x)^(i-1); A*A = 2^(n-1) I
    a = np.zeros((n, n))
    for i in range(n):
        p = np.array([1.0])
        plus = np.array([comb(n - 1 - i, k) for k in range(n - i)], dtype=float)
        minus = np.array([(-1.0) ** k * comb(i, k) for k in range(i + 1)], dtype=float)
        p = np.convolve(plus, minus)
        a[i, :p.size] = p
    return a


def invol(interp, args):
    n = _int(args[0], "n")
    i, j = _ij(n)
    a = 1.0 / (i + j - 1)
    d = -float(n)
    a[:, 0] *= d
    for k in range(1, n):
        d = -(n + k) * (n - k) * d / (k * k)
        a[k, :] *= d
    return a


def krylov(interp, args):
    rand, randn = _rng(interp)
    first = fnum(args[0])
    if first.size == 1:
        n = _int(args[0], "n")
        a = randn(n, n)
    else:
        a = first
        n = a.shape[0]
    x = fnum(_opt(args, 1, np.ones((n, 1)))).reshape(-1, 1)
    j = _int(_opt(args, 2, np.array([[float(n)]])), "j")
    b = np.ones((n, j), dtype=np.result_type(a.dtype, x.dtype, float))
    b[:, 0:1] = x
    for k in range(1, j):
        b[:, k] = a @ b[:, k - 1]
    return simplify(b)


def chebspec(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[0.0]])), "k")
    if k not in (0, 1):
        raise MatlabError("gallery('chebspec'): k must be 0 or 1.")
    if k == 1:
        n += 1
    m = n - 1
    x = np.cos(np.arange(m + 1) * np.pi / m)
    c = np.ones(m + 1)
    c[0] = c[m] = 2.0
    c = c * (-1.0) ** np.arange(m + 1)
    xx = np.tile(x.reshape(-1, 1), (1, m + 1))
    dx = xx - xx.T
    d = np.outer(c, 1.0 / c) / (dx + np.eye(m + 1))
    d = d - np.diag(d.sum(axis=1))
    if k == 1:
        d = d[1:, 1:]
    return d


def chebvand(interp, args):
    if len(args) == 1:
        p = _vec_or_n(args[0], "p")
        if fnum(args[0]).size == 1:
            p = np.linspace(0.0, 1.0, int(p.size))
        m = p.size
    else:
        m = _int(args[0], "m")
        p = _vec_or_n(args[1], "p")
        if fnum(args[1]).size == 1:
            p = np.linspace(0.0, 1.0, int(p.size))
    n = p.size
    c = np.ones((m, n), dtype=p.dtype)
    if m > 1:
        c[1, :] = p
    for i in range(2, m):
        c[i, :] = 2 * p * c[i - 1, :] - c[i - 2, :]
    return c


def cycol(interp, args):
    rand, randn = _rng(interp)
    nv = fnum(args[0]).flatten()
    m, n = int(nv[0]), int(nv[-1])
    k = _int(_opt(args, 1, np.array([[max(int(round(n / 4)), 1)]])), "k")
    a = randn(m, k)
    reps = int(np.ceil(n / k))
    return np.tile(a, (1, reps))[:, :n]


def qmult_matrix(interp, a: np.ndarray) -> np.ndarray:
    """Multiply by a random orthogonal matrix (Haar distributed) from the left."""
    _, randn = _rng(interp)
    n = a.shape[0]
    z = randn(n, n)
    q, r = np.linalg.qr(z)
    q = q * np.sign(np.where(np.diag(r) == 0, 1.0, np.diag(r)))
    return q @ a


def qmult(interp, args):
    first = fnum(args[0])
    a = np.eye(_int(args[0], "n")) if first.size == 1 else first
    return qmult_matrix(interp, a)


def randsvd(interp, args):
    rand, _ = _rng(interp)
    nv = fnum(args[0]).flatten()
    m, n = int(nv[0]), int(nv[-1])
    p = min(m, n)
    kappa = _num(_opt(args, 1, np.array([[np.sqrt(1.0 / _EPS)]])), "kappa")
    mode = _int(_opt(args, 2, np.array([[3.0]])), "mode")
    kl = _opt(args, 3)
    ku = _opt(args, 4)
    posdef = kappa < 0  # a negative kappa asks for a symmetric positive definite matrix
    kappa = abs(kappa)
    if kappa < 1:
        raise MatlabError("gallery('randsvd'): the condition number must be at least 1 in magnitude.")
    kind = abs(mode)
    if p == 1:
        sigma = np.ones(1)
    elif kind == 1:
        sigma = np.concatenate([[1.0], np.full(p - 1, 1.0 / kappa)])
    elif kind == 2:
        sigma = np.concatenate([np.ones(p - 1), [1.0 / kappa]])
    elif kind == 3:
        sigma = kappa ** (-np.arange(p) / (p - 1.0))
    elif kind == 4:
        sigma = 1.0 - np.arange(p) / (p - 1.0) * (1.0 - 1.0 / kappa)
    elif kind == 5:
        sigma = np.sort(np.exp(-rand(1, p).flatten() * np.log(kappa)))[::-1]
        sigma[0], sigma[-1] = 1.0, 1.0 / kappa
    else:
        raise MatlabError("gallery('randsvd'): mode must be 1, 2, 3, 4 or 5 (negative for the reverse order).")
    if mode < 0:
        sigma = sigma[::-1]
    if posdef:
        if m != n:
            raise MatlabError("gallery('randsvd'): a positive definite matrix (negative kappa) must be square.")
        q = qmult_matrix(interp, np.eye(n))
        a = (q.T * sigma) @ q
        return (a + a.T) / 2
    s = np.zeros((m, n))
    s[np.arange(p), np.arange(p)] = sigma
    a = qmult_matrix(interp, s)
    a = qmult_matrix(interp, a.T).T
    if kl is not None:
        lo = _int(kl, "kl")
        up = _int(ku, "ku") if ku is not None else lo
        # reduce to band form with orthogonal transformations (keeps the singular values)
        a = _bandred(a, lo, up)
    return a


def _bandred(a: np.ndarray, kl: int, ku: int) -> np.ndarray:
    """Two-sided Householder reduction to lower bandwidth kl and upper bandwidth ku."""
    a = a.copy()
    m, n = a.shape

    def reflect(x):
        v = x.copy()
        nrm = np.linalg.norm(x)
        if nrm == 0:
            return v, 0.0
        s = -np.copysign(nrm, x[0]) if x[0] != 0 else -nrm
        v[0] = x[0] - s
        return v, -1.0 / (s * v[0])

    for j in range(min(min(m, n), max(m - kl - 1, n - ku - 1))):
        if j + kl + 1 <= m - 1:
            v, beta = reflect(a[j + kl:, j])
            if beta:
                a[j + kl:, j:] -= beta * np.outer(v, v @ a[j + kl:, j:])
            a[j + kl + 1:, j] = 0.0
        if j + ku + 1 <= n - 1:
            v, beta = reflect(a[j, j + ku:])
            if beta:
                a[j:, j + ku:] -= beta * np.outer(a[j:, j + ku:] @ v, v)
            a[j, j + ku + 1:] = 0.0
    return a


def rando(interp, args):
    rand, _ = _rng(interp)
    nv = fnum(args[0]).flatten()
    m, n = int(nv[0]), int(nv[-1])
    k = _int(_opt(args, 1, np.array([[1.0]])), "k")
    u = rand(m, n)
    if k == 1:
        return np.floor(u + 0.5)
    if k == 2:
        return 2 * np.floor(u + 0.5) - 1
    if k == 3:
        return np.round(3 * u - 1.5) + 0.0
    raise MatlabError("gallery('rando'): k must be 1, 2 or 3.")


def randhess(interp, args):
    rand, _ = _rng(interp)
    first = fnum(args[0])
    if first.size == 1:
        n = _int(args[0], "n")
        x = rand(1, n).flatten() * 2 * np.pi
    else:
        x = first.flatten()
        n = x.size
    h = np.eye(n)
    h[n - 1, n - 1] = np.sign(x[n - 1]) if x[n - 1] != 0 else 1.0
    for i in range(n - 1, 0, -1):
        c, s = np.cos(x[i - 1]), np.sin(x[i - 1])
        rows = h[[i - 1, i], :].copy()
        h[i - 1, :] = c * rows[0] + s * rows[1]
        h[i, :] = -s * rows[0] + c * rows[1]
    return h


def randcolu(interp, args):
    rand, _ = _rng(interp)
    first = fnum(args[0])
    if first.size == 1:
        n = _int(args[0], "n")
        m = _int(_opt(args, 1, np.array([[float(n)]])), "m")
        x = rand(n, 1).flatten()
        x = np.sqrt(n) * x / np.linalg.norm(x)
    else:
        x = first.flatten()
        n = x.size
        m = _int(_opt(args, 1, np.array([[float(n)]])), "m")
    if m < n:
        raise MatlabError("gallery('randcolu'): m must be at least n.")
    if abs(np.sum(x ** 2) - n) > 100 * n * _EPS:
        raise MatlabError("gallery('randcolu'): the squares of the singular values must add up to n.")
    a = np.zeros((m, n))
    a[np.arange(n), np.arange(n)] = x
    a = qmult_matrix(interp, a)
    # rotate pairs of columns until every column has unit norm (the singular values do not change)
    a = _unit_columns(a)
    return a


def _unit_columns(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    n = a.shape[1]
    for _ in range(n):
        norms = np.sum(a * a, axis=0)
        lo = np.nonzero(norms < 1 - 1e-13)[0]
        hi = np.nonzero(norms > 1 + 1e-13)[0]
        if lo.size == 0 or hi.size == 0:
            break
        i, j = int(lo[0]), int(hi[0])
        aij = float(a[:, i] @ a[:, j])
        # choose t = tan(angle) so that the rotated column i has unit norm
        # (norms[i] + 2 t aij + t^2 norms[j]) / (1 + t^2) = 1
        qa, qb, qc = norms[j] - 1.0, 2.0 * aij, norms[i] - 1.0
        disc = max(qb * qb - 4 * qa * qc, 0.0)
        t = (-qb + np.sqrt(disc)) / (2 * qa)
        c = 1.0 / np.sqrt(1 + t * t)
        s = t * c
        ci, cj = a[:, i].copy(), a[:, j].copy()
        a[:, i] = c * ci + s * cj
        a[:, j] = -s * ci + c * cj
    return a


def randcorr(interp, args):
    rand, _ = _rng(interp)
    first = fnum(args[0])
    if first.size == 1:
        n = _int(args[0], "n")
        x = rand(n, 1).flatten()
        x = n * x / np.sum(x)
    else:
        x = first.flatten()
        n = x.size
    if np.any(x < 0) or abs(np.sum(x) - n) > 100 * n * _EPS:
        raise MatlabError("gallery('randcorr'): the eigenvalues must be nonnegative and add up to n.")
    a = np.zeros((n, n))
    a[np.arange(n), np.arange(n)] = np.sqrt(x)
    a = _unit_columns(qmult_matrix(interp, a))
    c = a.T @ a
    c = (c + c.T) / 2
    c[np.arange(n), np.arange(n)] = 1.0
    return c


def randjorth(interp, args):
    rand, _ = _rng(interp)
    p = _int(args[0], "p")
    q = _int(_opt(args, 1, np.array([[float(p)]])), "q")
    c = _num(_opt(args, 2, np.array([[np.sqrt(1.0 / _EPS)]])), "c")
    if p < 1 or q < 1 or c < 1:
        raise MatlabError("gallery('randjorth'): p and q must be positive and the condition number at least 1.")
    # hyperbolic CS form: blkdiag(U1, U2) * [C S; S C] * blkdiag(V1, V2) is J-orthogonal for
    # J = blkdiag(eye(p), -eye(q)); its singular values are the numbers e^t and e^-t below.
    m = min(p, q)
    sig = np.sqrt(c)
    if m == 1:
        cs = np.array([sig])
    else:
        mid = np.exp((2 * rand(1, m - 2).flatten() - 1) * np.log(sig)) if m > 2 else np.zeros(0)
        cs = np.concatenate([[1.0 / sig], mid, [sig]])
    ch = (cs + 1.0 / cs) / 2
    sh = (cs - 1.0 / cs) / 2
    core = np.eye(p + q)
    k = np.arange(m)
    core[k, k] = ch
    core[p + k, p + k] = ch
    core[k, p + k] = sh
    core[p + k, k] = sh

    def blk():
        z = np.zeros((p + q, p + q))
        z[:p, :p] = qmult_matrix(interp, np.eye(p))
        z[p:, p:] = qmult_matrix(interp, np.eye(q))
        return z

    return blk() @ core @ blk()


def wathen(interp, args):
    import scipy.sparse as sp
    rand, _ = _rng(interp)
    nx, ny = _int(args[0], "nx"), _int(args[1], "ny")
    k = _int(_opt(args, 2, np.array([[0.0]])), "k")
    e1 = np.array([[6, -6, 2, -8], [-6, 32, -6, 20], [2, -6, 6, -6], [-8, 20, -6, 32]], dtype=float)
    e2 = np.array([[3, -8, 2, -6], [-8, 16, -8, 20], [2, -8, 3, -8], [-6, 20, -8, 16]], dtype=float)
    em = np.block([[e1, e2], [e2.T, e1]]) / 45.0
    n = 3 * nx * ny + 2 * nx + 2 * ny + 1
    rho = 100 * rand(nx, ny)
    rows, cols, vals = [], [], []
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            nn = np.zeros(8, dtype=int)
            nn[0] = 3 * j * nx + 2 * i + 2 * j + 1
            nn[1] = nn[0] - 1
            nn[2] = nn[1] - 1
            nn[3] = (3 * j - 1) * nx + 2 * j + i - 1
            nn[4] = 3 * (j - 1) * nx + 2 * i + 2 * j - 3
            nn[5] = nn[4] + 1
            nn[6] = nn[5] + 1
            nn[7] = nn[3] + 1
            block = rho[i - 1, j - 1] * em
            for a in range(8):
                for b in range(8):
                    rows.append(nn[a] - 1)
                    cols.append(nn[b] - 1)
                    vals.append(block[a, b])
    m = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    if k == 1:
        d = 1.0 / np.sqrt(m.diagonal())
        m = sp.diags(d) @ m @ sp.diags(d)
    return _sparse(m)


def dorr(interp, args, nargout):
    import scipy.sparse as sp
    n = _int(args[0], "n")
    theta = _num(_opt(args, 1, np.array([[0.01]])), "theta")
    c = np.zeros(n)
    e = np.zeros(n)
    d = np.zeros(n)
    h = 1.0 / (n + 1)
    m = (n + 1) // 2
    term = theta / h ** 2
    i = np.arange(1, m + 1)
    c[:m] = -term
    e[:m] = c[:m] - (0.5 - i * h) / h
    d[:m] = -(c[:m] + e[:m])
    i = np.arange(m + 1, n + 1)
    e[m:] = -term
    c[m:] = e[m:] + (0.5 - i * h) / h
    d[m:] = -(c[m:] + e[m:])
    c, e = c[1:], e[:-1]
    if nargout > 1:
        return c.reshape(-1, 1), d.reshape(-1, 1), e.reshape(-1, 1)
    return _sparse(sp.diags([c, d, e], [-1, 0, 1], shape=(n, n), format="csc"))


def condex(interp, args):
    n = _int(args[0], "n")
    k = _int(_opt(args, 1, np.array([[4.0]])), "k")
    theta = _num(_opt(args, 2, np.array([[100.0]])), "theta")
    if k == 1:  # Cline and Rew, 4-by-4
        a = np.array([[1, -1, -2 * theta, 0], [0, 1, theta, -theta], [0, 1, 1 + theta, -(theta + 1)],
                      [0, 0, 0, theta]], dtype=float)
    elif k == 2:  # Cline and Rew, 3-by-3
        a = np.array([[1, 1 - 2 / theta ** 2, -2], [0, 1 / theta, -1 / theta], [0, 0, 1]], dtype=float)
    elif k == 3:  # Cline, Moler, Stewart and Wilkinson: any n
        a = np.eye(n) - np.tril(np.ones((n, n)), -1)
        if n:
            a[n - 1, n - 1] = -1.0
        return a
    elif k == 4:  # Higham: the identity plus theta times a projector, any n >= 3
        x = np.ones((n, 3))
        x[1:, 0] = 0.0
        x[:, 1] = (-1.0) ** np.arange(n) * (1 + np.arange(n) / (n - 1.0))
        q, _ = np.linalg.qr(x)
        return np.eye(n) + theta * (np.eye(n) - q @ q.T)
    else:
        raise MatlabError("gallery('condex'): k must be 1, 2, 3 or 4.")
    base = a.shape[0]
    if n > base:  # embed the small matrix in an identity
        out = np.eye(n)
        out[:base, :base] = a
        return out
    return a


def _seeded(seed: int):
    return np.random.RandomState(int(seed) % (2 ** 32))


def _dims_and_class(args: list, name: str):
    """Trailing size arguments, an optional class name and a required seed (the last number)."""
    cls = "double"
    rest = list(args)
    if rest and is_text(rest[-1]):
        cls = text(rest.pop())
        if cls not in ("double", "single"):
            raise MatlabError(f"gallery('{name}'): the class must be 'double' or 'single'.")
    if len(rest) < 2:
        raise MatlabError(f"gallery('{name}'): expected the size and a seed.")
    seed = _int(rest[-1], "the seed")
    dims: list[int] = []
    for a in rest[:-1]:
        dims.extend(int(x) for x in np.real(fnum(a)).flatten())
    if len(dims) == 1:
        dims = [dims[0], dims[0]]
    return tuple(max(d, 0) for d in dims), seed, cls


def uniformdata(interp, args):
    dims, seed, cls = _dims_and_class(args, "uniformdata")
    a = _seeded(seed).rand(int(np.prod(dims))).reshape(dims, order="F")
    return a.astype(np.float32) if cls == "single" else a


def normaldata(interp, args):
    dims, seed, cls = _dims_and_class(args, "normaldata")
    a = _seeded(seed).randn(int(np.prod(dims))).reshape(dims, order="F")
    return a.astype(np.float32) if cls == "single" else a


def integerdata(interp, args):
    cls = "double"
    rest = list(args)
    if rest and is_text(rest[-1]):
        cls = text(rest.pop())
    if len(rest) < 3:
        raise MatlabError("gallery('integerdata'): expected the range, the size and a seed.")
    rng = np.real(fnum(rest[0])).flatten()
    lo, hi = (1, int(rng[0])) if rng.size == 1 else (int(rng[0]), int(rng[1]))
    seed = _int(rest[-1], "the seed")
    dims: list[int] = []
    for a in rest[1:-1]:
        dims.extend(int(x) for x in np.real(fnum(a)).flatten())
    if len(dims) == 1:
        dims = [dims[0], dims[0]]
    a = _seeded(seed).randint(lo, hi + 1, size=int(np.prod(dims))).reshape(dims, order="F").astype(float)
    from ._util import cast_class
    return cast_class(a, cls)


FAMILIES = {
    "binomial": binomial, "cauchy": cauchy, "chebspec": chebspec, "chebvand": chebvand, "chow": chow,
    "circul": circul, "clement": clement, "condex": condex, "cycol": cycol, "dorr": dorr,
    "dramadah": dramadah, "fiedler": fiedler, "forsythe": forsythe, "frank": frank, "gcdmat": gcdmat,
    "gearmat": gearmat, "grcar": grcar, "hanowa": hanowa, "house": house, "integerdata": integerdata,
    "invhess": invhess, "invol": invol, "ipjfact": ipjfact, "jordbloc": jordbloc, "kahan": kahan,
    "kms": kms, "krylov": krylov, "lauchli": lauchli, "lehmer": lehmer, "lesp": lesp, "lotkin": lotkin,
    "minij": minij, "moler": moler, "neumann": neumann, "normaldata": normaldata, "orthog": orthog,
    "parter": parter, "pei": pei, "poisson": poisson, "prolate": prolate, "qmult": qmult,
    "randcolu": randcolu, "randcorr": randcorr, "randhess": randhess, "randjorth": randjorth,
    "rando": rando, "randsvd": randsvd, "redheff": redheff, "riemann": riemann, "ris": ris,
    "smoke": smoke, "toeppd": toeppd, "toeppen": toeppen, "tridiag": tridiag, "triw": triw,
    "uniformdata": uniformdata, "wathen": wathen, "wilk": wilk,
}
MULTI_OUTPUT = {"house", "wilk", "ipjfact", "dorr"}
