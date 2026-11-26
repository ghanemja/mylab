"""Numerical core of the control-system functions: polynomials and state-space matrices.

Everything here works on plain NumPy arrays and knows nothing about the interpreter, so
the model classes in :mod:`control` and the analysis and design functions share it.
SciPy is imported inside the functions that need it.
"""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError

EPS = float(np.finfo(float).eps)
IMPROPER_ID = "Control:ltiobject:improper"


# -- polynomials (coefficient vectors, highest power first) -----------------------------------

def ptrim(p) -> np.ndarray:
    """Remove leading zeros; the zero polynomial becomes [0]."""
    p = np.atleast_1d(np.asarray(p))
    if p.dtype.kind not in "fc":
        p = p.astype(float)
    p = p.ravel()
    nz = np.flatnonzero(p != 0)
    if nz.size == 0:
        return np.zeros(1, dtype=p.dtype)
    return p[nz[0]:].copy()


def pdeg(p) -> int:
    return len(ptrim(p)) - 1


def piszero(p) -> bool:
    return not np.any(np.asarray(p) != 0)


def padd(a, b) -> np.ndarray:
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    n = max(a.size, b.size)
    out = np.zeros(n, dtype=np.result_type(a, b, float))
    out[n - a.size:] += a
    out[n - b.size:] += b
    return ptrim(out)


def pder(p) -> np.ndarray:
    """Derivative of a polynomial; the derivative of a constant is [0]."""
    p = np.atleast_1d(p)
    if p.size <= 1:
        return np.zeros(1)
    return p[:-1] * np.arange(p.size - 1, 0, -1)


def pmul(a, b) -> np.ndarray:
    return ptrim(np.convolve(np.atleast_1d(a), np.atleast_1d(b)))


def peq(a, b) -> bool:
    a, b = ptrim(a), ptrim(b)
    return a.size == b.size and bool(np.all(a == b))


def proots(p) -> np.ndarray:
    p = ptrim(p)
    if p.size <= 1:
        return np.zeros(0)
    return np.roots(p)


def real_if_close(x, tol: float = 1e3) -> np.ndarray:
    """Drop imaginary parts that are rounding noise (relative to the largest entry)."""
    x = np.asarray(x)
    if x.dtype.kind != "c":
        return x
    if x.size == 0:
        return x.real.copy()
    scale = max(float(np.max(np.abs(x))), 1e-300)
    if float(np.max(np.abs(x.imag))) <= tol * EPS * scale:
        return x.real.copy()
    return x


def pair_conjugates(r, tol: float = 1e-8) -> np.ndarray:
    """Make nearly conjugate pairs exactly conjugate and nearly real values real."""
    r = np.array(r, dtype=complex).ravel()
    if r.size == 0:
        return r.real.copy()
    out = r.copy()
    scale = np.maximum(np.abs(r), 1.0)
    used = np.zeros(r.size, dtype=bool)
    for i in range(r.size):
        if used[i]:
            continue
        if abs(r[i].imag) <= tol * scale[i]:
            continue
        best, bd = -1, np.inf
        for j in range(r.size):
            if j == i or used[j]:
                continue
            d = abs(r[j] - np.conj(r[i]))
            if d < bd:
                best, bd = j, d
        if best >= 0 and bd <= tol * scale[i]:
            avg = (r[i] + np.conj(r[best])) / 2
            out[i], out[best] = avg, np.conj(avg)
            used[i] = used[best] = True
    lone = ~used & (np.abs(out.imag) <= tol * scale)
    out[lone] = out[lone].real
    if not np.any(out.imag):
        return out.real.copy()
    return out


def rpoly(roots) -> np.ndarray:
    """Monic polynomial with the given roots; real when the roots are real or conjugate pairs."""
    roots = np.asarray(roots).ravel()
    if roots.size == 0:
        return np.ones(1)
    p = np.poly(roots)
    if np.iscomplexobj(p):
        p = np.poly(pair_conjugates(roots))
        if np.iscomplexobj(p):
            p = real_if_close(p, 1e4)
    return np.asarray(p)


def cplxpair_sorted(r) -> np.ndarray:
    """Roots in a stable display order: by real part, conjugate pairs adjacent (positive imag first)."""
    r = np.asarray(r).ravel()
    if r.size == 0:
        return r
    keys = sorted(range(r.size), key=lambda i: (round(float(np.real(r[i])), 12), abs(float(np.imag(r[i]))),
                                                -float(np.imag(r[i]))))
    return r[keys]


# -- matrices ---------------------------------------------------------------------------------

def mat(a, what: str = "matrix") -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind not in "fc":
        a = a.astype(float)
    if a.ndim < 2:
        a = np.atleast_2d(a)
    if a.ndim != 2:
        raise MatlabError(f"The {what} must be a 2-D matrix.")
    return a


def blkdiag(*ms) -> np.ndarray:
    ms = [np.atleast_2d(m) for m in ms]
    r = sum(m.shape[0] for m in ms)
    c = sum(m.shape[1] for m in ms)
    out = np.zeros((r, c), dtype=np.result_type(*ms, float) if ms else float)
    i = j = 0
    for m in ms:
        out[i:i + m.shape[0], j:j + m.shape[1]] = m
        i += m.shape[0]
        j += m.shape[1]
    return out


def solve(a, b) -> np.ndarray:
    a = np.atleast_2d(a)
    if a.shape[0] == 0:
        return np.zeros((0,) + np.shape(b)[1:])
    return np.linalg.solve(a, b)


def expm(a) -> np.ndarray:
    a = np.atleast_2d(a)
    if a.shape[0] == 0:
        return np.zeros((0, 0))
    if a.shape == (1, 1):
        return np.exp(a)
    from scipy.linalg import expm as _expm
    return _expm(a)


def eigvals(a) -> np.ndarray:
    a = np.atleast_2d(a)
    if a.shape[0] == 0:
        return np.zeros(0)
    return np.linalg.eigvals(a)


def charpoly(a) -> np.ndarray:
    """Characteristic polynomial det(sI - A)."""
    a = np.atleast_2d(a)
    n = a.shape[0]
    if n == 0:
        return np.ones(1)
    if n <= 10:
        return _leverrier(a)[0]
    return rpoly(np.linalg.eigvals(a)) if not np.iscomplexobj(a) else np.poly(np.linalg.eigvals(a))


def _leverrier(a):
    """Faddeev-LeVerrier: coefficients of det(sI - A) and the matrices of adj(sI - A)."""
    n = a.shape[0]
    coeffs = np.zeros(n + 1, dtype=a.dtype if a.dtype.kind == "c" else float)
    coeffs[0] = 1.0
    mats = [np.eye(n, dtype=coeffs.dtype)]
    for k in range(1, n + 1):
        am = a @ mats[-1]
        coeffs[k] = -np.trace(am) / k
        if k < n:
            mats.append(am + coeffs[k] * np.eye(n))
    return coeffs, mats


def ss_numden(a, b, c, d):
    """Numerator coefficients (ny, nu, n+1) and the common denominator of C (sI-A)^-1 B + D."""
    a, b, c, d = (np.atleast_2d(m) for m in (a, b, c, d))
    n = a.shape[0]
    ny, nu = d.shape
    if n == 0:
        return d.reshape(ny, nu, 1).astype(float if d.dtype.kind != "c" else complex), np.ones(1)
    if n <= 10:
        den, mats = _leverrier(a)
        num = np.zeros((ny, nu, n + 1), dtype=np.result_type(den, b, c, d))
        for k, m in enumerate(mats):  # adj(sI - A) = sum_k s^(n-1-k) N_k
            num[:, :, k + 1] = c @ m @ b
        num += d[:, :, None] * den[None, None, :]
        return num, den
    den = charpoly(a)
    num = np.zeros((ny, nu, n + 1), dtype=np.result_type(den, b, c, d))
    for i in range(ny):
        for j in range(nu):
            z, k = siso_zeros_gain(a, b[:, [j]], c[[i], :], d[i, j])
            pz = k * rpoly(z) if k != 0 else np.zeros(1)
            num[i, j, n + 1 - pz.size:] = pz
    return num, den


# -- canonical realizations -------------------------------------------------------------------

def tf2ss_siso(num, den):
    """Controller canonical form of a proper transfer function num/den."""
    num, den = ptrim(num), ptrim(den)
    if piszero(den):
        raise MatlabError("The denominator of a transfer function cannot be zero.")
    n = den.size - 1
    if num.size - 1 > n:
        raise MatlabError("An improper transfer function (more zeros than poles) has no state-space "
                          "realization here; multiply it with other blocks first or add a fast pole.",
                          IMPROPER_ID)
    dtype = np.result_type(num, den, float)
    num = np.concatenate([np.zeros(n + 1 - num.size, dtype=dtype), num]) / den[0]
    den = den / den[0]
    a = np.zeros((n, n), dtype=dtype)
    b = np.zeros((n, 1), dtype=dtype)
    c = np.zeros((1, n), dtype=dtype)
    dd = np.array([[num[0]]], dtype=dtype)
    if n:
        a[0, :] = -den[1:]
        if n > 1:
            a[1:, :-1] = np.eye(n - 1)
        b[0, 0] = 1.0
        c[0, :] = num[1:] - num[0] * den[1:]
    return a, b, c, dd


def ss_mul(s1, s2):
    """Series connection: the system s1*s2 (s2 acts first), states ordered [x1; x2]."""
    a1, b1, c1, d1 = s1
    a2, b2, c2, d2 = s2
    n1, n2 = a1.shape[0], a2.shape[0]
    a = np.block([[a1, b1 @ c2], [np.zeros((n2, n1)), a2]])
    b = np.vstack([b1 @ d2, b2])
    c = np.hstack([c1, d1 @ c2])
    return a, b, c, d1 @ d2


def ss_add(s1, s2, sign: float = 1.0):
    a1, b1, c1, d1 = s1
    a2, b2, c2, d2 = s2
    return blkdiag(a1, a2), np.vstack([b1, b2]), np.hstack([c1, sign * c2]), d1 + sign * d2


def ss_inv(s):
    a, b, c, d = s
    if d.shape[0] != d.shape[1]:
        raise MatlabError("Only square systems can be inverted.")
    if d.shape[0] and (np.linalg.matrix_rank(d) < d.shape[0]):
        raise MatlabError("Cannot invert this model: the feedthrough matrix D is singular, so the inverse "
                          "is not proper.")
    di = np.linalg.inv(d) if d.size else d.copy()
    return a - b @ di @ c, b @ di, -di @ c, di


def ss_append(*systems):
    a = blkdiag(*[s[0] for s in systems])
    b = blkdiag(*[s[1] for s in systems])
    c = blkdiag(*[s[2] for s in systems])
    d = blkdiag(*[s[3] for s in systems])
    return a, b, c, d


def ss_feedback(s1, s2, sign: float = -1.0):
    """Closed loop of s1 with s2 in the return path: y = s1 (u + sign * s2 y)."""
    a1, b1, c1, d1 = s1
    a2, b2, c2, d2 = s2
    ny = d1.shape[0]
    m = np.eye(ny) - sign * d1 @ d2
    if ny and np.linalg.cond(m) > 1 / EPS:
        raise MatlabError("The feedback loop is algebraic and singular (I - D1*D2 cannot be inverted).")
    f = np.linalg.inv(m) if ny else m
    fc1, fd1 = f @ c1, f @ d1
    # e = u + sign * (C2 x2 + D2 y1)
    ex1 = sign * d2 @ fc1
    ex2 = sign * (c2 + d2 @ (sign * fd1 @ c2))
    eu = np.eye(d1.shape[1]) + sign * d2 @ fd1
    a = np.block([[a1 + b1 @ ex1, b1 @ ex2], [b2 @ fc1, a2 + b2 @ (sign * fd1 @ c2)]])
    b = np.vstack([b1 @ eu, b2 @ fd1])
    c = np.hstack([fc1, sign * fd1 @ c2])
    return a, b, c, fd1


def ss_interconnect(s, k, ein, eout):
    """Close the static loop u = K y + Ein r around s and read the outputs Eout y."""
    a, b, c, d = s
    ny = d.shape[0]
    m = np.eye(ny) - d @ k
    if ny and np.linalg.cond(m) > 1 / EPS:
        raise MatlabError("The interconnection has an algebraic loop that cannot be solved.")
    f = np.linalg.inv(m) if ny else m
    fc, fd = f @ c, f @ d @ ein
    return a + b @ k @ fc, b @ (k @ fd + ein), eout @ fc, eout @ fd


def ss_lft(s1, s2, nu: int, ny: int):
    """Lower LFT: the last nu inputs / ny outputs of s1 are closed with s2 (u2 = s2 y2, positive)."""
    a1, b1, c1, d1 = s1
    a2, b2, c2, d2 = s2
    n1, n2 = a1.shape[0], a2.shape[0]
    p1, m1 = d1.shape
    if d2.shape != (nu, ny) or nu > m1 or ny > p1:
        raise MatlabError("lft: the dimensions of the two systems do not match.")
    sysa = ss_append(s1, s2)
    nyt, nut = p1 + nu, m1 + ny
    k = np.zeros((nut, nyt))
    # inputs of s1 (last nu) come from outputs of s2; inputs of s2 from last ny outputs of s1
    k[m1 - nu:m1, p1:] = np.eye(nu)
    k[m1:, p1 - ny:p1] = np.eye(ny)
    ein = np.zeros((nut, m1 - nu))
    ein[:m1 - nu, :] = np.eye(m1 - nu)
    eout = np.zeros((p1 - ny, nyt))
    eout[:, :p1 - ny] = np.eye(p1 - ny)
    del n1, n2
    return ss_interconnect(sysa, k, ein, eout)


# -- frequency response -----------------------------------------------------------------------

def ss_freqresp(a, b, c, d, s) -> np.ndarray:
    """C (sI - A)^-1 B + D for each complex point in ``s``: array (ny, nu, len(s))."""
    s = np.atleast_1d(np.asarray(s, dtype=complex)).ravel()
    n = a.shape[0]
    ny, nu = d.shape
    out = np.empty((ny, nu, s.size), dtype=complex)
    if n == 0 or ny == 0 or nu == 0:
        out[:] = d[:, :, None]
        return out
    eye = np.eye(n)
    with np.errstate(all="ignore"):
        for k, sk in enumerate(s):
            if not np.isfinite(sk):
                out[:, :, k] = d
                continue
            try:
                x = np.linalg.solve(sk * eye - a, b)
                out[:, :, k] = c @ x + d
            except np.linalg.LinAlgError:
                out[:, :, k] = np.inf
    return out


def poly_freqresp(num, den, s) -> np.ndarray:
    s = np.asarray(s, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.polyval(num, s)
        dd = np.polyval(den, s)
        out = n / dd
        inf_pts = ~np.isfinite(s)
        if np.any(inf_pts):
            dn, ddg = pdeg(num), pdeg(den)
            lim = 0.0 if dn < ddg else (ptrim(num)[0] / ptrim(den)[0] if dn == ddg else np.inf)
            out = np.where(inf_pts, lim, out)
        bad = (dd == 0) & (n != 0)
        out = np.where(bad, np.inf, out)
    return out


def zpk_freqresp(z, p, k, s) -> np.ndarray:
    s = np.asarray(s, dtype=complex)
    out = np.full(s.shape, complex(k))
    with np.errstate(divide="ignore", invalid="ignore"):
        zz, pp = list(z), list(p)
        # interleave factors so that large orders neither overflow nor underflow
        for i in range(max(len(zz), len(pp))):
            if i < len(zz):
                out = out * (s - zz[i])
            if i < len(pp):
                out = out / (s - pp[i])
    return out


# -- zeros ------------------------------------------------------------------------------------

def _finite_geig(m, nmat):
    from scipy.linalg import eig
    w = eig(m, nmat, right=False, homogeneous_eigvals=True)
    alpha, beta = w[0], w[1]
    scale = max(float(np.linalg.norm(m, 1)), 1.0)
    keep = np.abs(beta) > 1e-11 * np.maximum(np.abs(alpha) / scale, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = alpha[keep] / beta[keep]
    return vals[np.isfinite(vals)]


def _balance_abcd(a, b, c):
    """Diagonal state scaling (powers of two) that evens out the row/column norms of [A B; C 0]."""
    n = a.shape[0]
    if n == 0:
        return a, b, c
    t = np.ones(n)
    aa = np.abs(a).astype(float)
    np.fill_diagonal(aa, 0.0)
    ab, ac = np.abs(b), np.abs(c)
    for _ in range(30):
        changed = False
        for i in range(n):
            # with x = T z the entry (i, j) of A becomes a_ij * t_j / t_i
            row = (np.sum(aa[i, :] * t) + np.sum(ab[i, :])) / t[i]
            col = (np.sum(aa[:, i] / t) + np.sum(ac[:, i])) * t[i]
            if row == 0 or col == 0:
                continue
            f = 2.0 ** np.round(0.5 * np.log2(row / col))
            if f != 1.0:
                t[i] *= f
                changed = True
        if not changed:
            break
    return a * t[None, :] / t[:, None], b / t[:, None], c * t[None, :]


def ss_tzero(a, b, c, d) -> np.ndarray:
    """Invariant (transmission) zeros of a state-space model."""
    a, b, c, d = (np.atleast_2d(np.asarray(m)) for m in (a, b, c, d))
    n = a.shape[0]
    ny, nu = d.shape
    if n == 0 or ny == 0 or nu == 0:
        return np.zeros(0)
    a, b, c = _balance_abcd(a, b, c)
    if ny == nu:
        z = _square_zeros(a, b, c, d)
    else:
        rng = np.random.default_rng(20240229)
        sets = []
        for _ in range(2):
            if ny > nu:
                k = rng.standard_normal((nu, ny))
                sets.append(_square_zeros(a, b, k @ c, k @ d))
            else:
                k = rng.standard_normal((nu, ny))
                sets.append(_square_zeros(a, b @ k, c, d @ k))
        z = _common(sets[0], sets[1])
    z = pair_conjugates(z) if not (np.iscomplexobj(a) or np.iscomplexobj(b) or np.iscomplexobj(c)
                                   or np.iscomplexobj(d)) else z
    return z


def _square_zeros(a, b, c, d) -> np.ndarray:
    n = a.shape[0]
    m = d.shape[0]
    big = np.block([[a, b], [c, d]])
    nm = blkdiag(np.eye(n), np.zeros((m, m)))
    # a pencil that is singular for every s has no (generic) zeros to report
    probe = 0.7310585 + 0.4142135j
    test = big - probe * nm
    if np.linalg.cond(test) > 1e13:
        return np.zeros(0)
    return _finite_geig(big, nm)


def _common(z1, z2, tol: float = 1e-6) -> np.ndarray:
    out = []
    used = np.zeros(len(z2), dtype=bool)
    for v in z1:
        if len(z2) == 0:
            break
        dist = np.abs(z2 - v)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= tol * max(1.0, abs(v)):
            used[j] = True
            out.append((v + z2[j]) / 2)
    return np.array(out, dtype=complex)


def siso_zeros_gain(a, b, c, d):
    """Zeros and leading gain k of the SISO transfer function k * prod(s - z) / det(sI - A)."""
    a, b, c = np.atleast_2d(a), np.atleast_2d(b).reshape(-1, 1), np.atleast_2d(c).reshape(1, -1)
    d = complex(np.asarray(d).ravel()[0]) if np.iscomplexobj(d) else float(np.asarray(d).ravel()[0])
    n = a.shape[0]
    if n == 0:
        return np.zeros(0), d
    if d != 0:
        z = eigvals(a - b @ c / d)  # zeros of a biproper system are the poles of its inverse
        z = pair_conjugates(z) if not np.iscomplexobj(a) else z
        return z, d
    # Markov parameters c A^k b: the first nonzero one is the gain, its index the relative degree
    scale = (np.linalg.norm(c) * np.linalg.norm(b)) or 1.0
    v = b.copy()
    an = max(np.linalg.norm(a, 2), 1e-300) if n else 1.0
    for r in range(1, n + 1):
        g = (c @ v).item()
        if abs(g) > 1e-10 * scale * an ** (r - 1):
            ab, bb, cb = _balance_abcd(a, b, c)
            z = _square_zeros(ab, bb, cb, np.zeros((1, 1)))
            nz = n - r
            if len(z) > nz:
                z = z[np.argsort(np.abs(z))[:nz]]
            z = pair_conjugates(z) if not np.iscomplexobj(a) else z
            return z, g
        v = a @ v
    return np.zeros(0), 0.0


# -- structural and numerical reduction -------------------------------------------------------

def sminreal_states(a, b, c) -> np.ndarray:
    """Indices of states that are structurally connected to both an input and an output."""
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    adj = a != 0
    reach = np.any(b != 0, axis=1)
    while True:  # x_i' depends on x_j when A[i, j] != 0
        new = reach | np.any(adj[:, reach], axis=1)
        if np.array_equal(new, reach):
            break
        reach = new
    obs = np.any(c != 0, axis=0)
    while True:
        new = obs | np.any(adj[obs, :], axis=0)
        if np.array_equal(new, obs):
            break
        obs = new
    return np.flatnonzero(reach & obs)


def ctrb_staircase(a, b, tol: float | None = None):
    """Orthogonal T with T A T' = [[Ac, A12], [0, Anc]] and T B = [Bc; 0]. Returns T, nc, block sizes."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    n = a.shape[0]
    if n == 0:
        return np.zeros((0, 0)), 0, []
    if tol is None:
        tol = n * max(np.linalg.norm(a, 1), np.linalg.norm(b, 1), 1e-300) * EPS * 10
    t = np.eye(n, dtype=np.result_type(a, b, float))
    aa, bb = a.astype(t.dtype).copy(), b.astype(t.dtype).copy()
    start, sizes = 0, []
    block = bb  # the part whose row space decides the next block
    prev = 0
    first = True
    while start < n:
        sub = block[start:, :] if first else aa[start:, prev:start]
        if sub.size == 0:
            break
        u, s, _ = np.linalg.svd(sub, full_matrices=True)
        r = int(np.sum(s > tol))
        if r == 0:
            break
        q = np.eye(n, dtype=t.dtype)
        q[start:, start:] = u.conj().T
        aa = q @ aa @ q.conj().T
        bb = q @ bb
        t = q @ t
        block = bb
        sizes.append(r)
        prev = start
        start += r
        first = False
    return t, start, sizes


def ss_minreal(a, b, c, d, tol: float | None = None):
    """Remove uncontrollable and unobservable states (orthogonal staircase reductions)."""
    a, b, c, d = (np.atleast_2d(m) for m in (a, b, c, d))
    n = a.shape[0]
    if n == 0:
        return a, b, c, d
    scale = max(np.linalg.norm(a, 1), np.linalg.norm(b, 1), np.linalg.norm(c, 1), 1e-300)
    eff = (np.sqrt(EPS) if tol is None else tol) * scale
    t, nc, _ = ctrb_staircase(a, b, eff)
    a1 = (t @ a @ t.conj().T)[:nc, :nc]
    b1 = (t @ b)[:nc, :]
    c1 = (c @ t.conj().T)[:, :nc]
    if nc == 0:
        return a1, b1, c1, d
    t2, no, _ = ctrb_staircase(a1.conj().T, c1.conj().T, eff)
    a2 = (t2 @ a1.conj().T @ t2.conj().T)[:no, :no].conj().T
    c2 = (t2 @ c1.conj().T)[:no, :].conj().T
    b2 = (b1.conj().T @ t2.conj().T)[:, :no].conj().T
    return a2, b2, c2, d


# -- Lyapunov, Sylvester and Riccati equations ------------------------------------------------

def _check_square(a, name="A"):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise MatlabError(f"{name} must be a square matrix.")


def lyap_c(a, q) -> np.ndarray:
    """X with A X + X A' + Q = 0."""
    a, q = np.atleast_2d(a), np.atleast_2d(q)
    _check_square(a)
    if q.shape != a.shape:
        raise MatlabError("lyap: A and Q must be square matrices of the same size.")
    if a.shape[0] == 0:
        return np.zeros((0, 0))
    from scipy.linalg import solve_continuous_lyapunov
    ev = np.linalg.eigvals(a)
    if np.min(np.abs(ev[:, None] + np.conj(ev)[None, :])) <= 1e-12 * max(1.0, float(np.max(np.abs(ev)))):
        raise MatlabError("The Lyapunov equation has no unique solution: A and -A' share an eigenvalue "
                          "(for example A has a pole on the imaginary axis).")
    return solve_continuous_lyapunov(a, -q)


def sylvester(a, b, c) -> np.ndarray:
    """X with A X + X B + C = 0."""
    a, b, c = np.atleast_2d(a), np.atleast_2d(b), np.atleast_2d(c)
    _check_square(a)
    _check_square(b, "B")
    if c.shape != (a.shape[0], b.shape[0]):
        raise MatlabError("lyap: C must have as many rows as A and as many columns as B.")
    from scipy.linalg import solve_sylvester
    try:
        return solve_sylvester(a, b, -c)
    except np.linalg.LinAlgError:
        raise MatlabError("The Sylvester equation has no unique solution (A and -B share an eigenvalue).") \
            from None


def lyap_d(a, q) -> np.ndarray:
    """X with A X A' - X + Q = 0."""
    a, q = np.atleast_2d(a), np.atleast_2d(q)
    _check_square(a)
    if q.shape != a.shape:
        raise MatlabError("dlyap: A and Q must be square matrices of the same size.")
    if a.shape[0] == 0:
        return np.zeros((0, 0))
    ev = np.linalg.eigvals(a)
    if np.min(np.abs(ev[:, None] * np.conj(ev)[None, :] - 1)) <= 1e-12:
        raise MatlabError("The discrete Lyapunov equation has no unique solution: two eigenvalues of A "
                          "multiply to 1 (for example a pole on the unit circle).")
    from scipy.linalg import solve_discrete_lyapunov
    return solve_discrete_lyapunov(a, q)


def sylvester_d(a, b, c) -> np.ndarray:
    """X with A X B - X + C = 0."""
    a, b, c = np.atleast_2d(a), np.atleast_2d(b), np.atleast_2d(c)
    n, m = a.shape[0], b.shape[0]
    if c.shape != (n, m):
        raise MatlabError("dlyap: C must have as many rows as A and as many columns as B.")
    k = np.kron(b.T, a) - np.eye(n * m)
    x = np.linalg.solve(k, -c.reshape(-1, order="F"))
    return x.reshape((n, m), order="F")


def sym(x) -> np.ndarray:
    return (x + x.conj().T) / 2


def care(a, b, q, r=None, s=None, e=None):
    """Stabilizing X of A'XE + E'XA - (E'XB + S) R^-1 (B'XE + S') + Q = 0, with gain and poles."""
    a, b, q = np.atleast_2d(a), np.atleast_2d(b), np.atleast_2d(q)
    n, m = a.shape[0], b.shape[1]
    r = np.eye(m) if r is None else np.atleast_2d(r)
    _riccati_check(a, b, q, r, s, e, "care")
    from scipy.linalg import solve_continuous_are
    try:
        x = solve_continuous_are(a, b, q, r, e=e, s=s)
    except (np.linalg.LinAlgError, ValueError) as err:
        raise MatlabError("The Riccati equation has no stabilizing solution: check that (A, B) is "
                          "stabilizable and that the weights are consistent (R positive definite). "
                          f"[{err}]") from None
    x = sym(x)
    ee = np.eye(n) if e is None else np.atleast_2d(e)
    ss_ = np.zeros((n, m)) if s is None else np.atleast_2d(s)
    g = np.linalg.solve(r, b.conj().T @ x @ ee + ss_.conj().T)
    from scipy.linalg import eigvals as geig
    lam = geig(a - b @ g, ee) if e is not None else np.linalg.eigvals(a - b @ g)
    if lam.size and np.max(lam.real) > 1e-6 * max(1.0, float(np.max(np.abs(lam)))):
        raise MatlabError("The Riccati equation has no stabilizing solution: check that (A, B) is "
                          "stabilizable and (Q, A) has no unobservable mode on the imaginary axis.")
    return x, g, lam


def dare(a, b, q, r=None, s=None, e=None):
    """Stabilizing X of A'XA - E'XE - (A'XB + S)(B'XB + R)^-1 (B'XA + S') + Q = 0."""
    a, b, q = np.atleast_2d(a), np.atleast_2d(b), np.atleast_2d(q)
    n, m = a.shape[0], b.shape[1]
    r = np.eye(m) if r is None else np.atleast_2d(r)
    _riccati_check(a, b, q, r, s, e, "dare")
    from scipy.linalg import solve_discrete_are
    try:
        x = solve_discrete_are(a, b, q, r, e=e, s=s)
    except (np.linalg.LinAlgError, ValueError) as err:
        raise MatlabError("The discrete Riccati equation has no stabilizing solution: check that (A, B) is "
                          f"stabilizable and that the weights are consistent. [{err}]") from None
    x = sym(x)
    ee = np.eye(n) if e is None else np.atleast_2d(e)
    ss_ = np.zeros((n, m)) if s is None else np.atleast_2d(s)
    g = np.linalg.solve(b.conj().T @ x @ b + r, b.conj().T @ x @ a + ss_.conj().T)
    from scipy.linalg import eigvals as geig
    lam = geig(a - b @ g, ee) if e is not None else np.linalg.eigvals(a - b @ g)
    if lam.size and np.max(np.abs(lam)) > 1 + 1e-6:
        raise MatlabError("The discrete Riccati equation has no stabilizing solution: check that (A, B) is "
                          "stabilizable and (Q, A) has no unobservable mode on the unit circle.")
    return x, g, lam


def _riccati_check(a, b, q, r, s, e, name):
    n, m = a.shape[0], b.shape[1]
    if a.shape != (n, n):
        raise MatlabError(f"{name}: A must be square.")
    if b.shape[0] != n:
        raise MatlabError(f"{name}: A and B must have the same number of rows.")
    if q.shape != (n, n):
        raise MatlabError(f"{name}: Q must be a square matrix of the same size as A.")
    if r.shape != (m, m):
        raise MatlabError(f"{name}: R must be a square matrix with one row per column of B.")
    if s is not None and np.atleast_2d(s).shape != (n, m):
        raise MatlabError(f"{name}: S must have one row per state and one column per input.")
    if e is not None and np.atleast_2d(e).shape != (n, n):
        raise MatlabError(f"{name}: E must be a square matrix of the same size as A.")


# -- discretization ---------------------------------------------------------------------------

def zoh_matrices(a, b, dt: float):
    """Phi = exp(A dt) and Gamma = integral_0^dt exp(A s) ds B."""
    n, m = a.shape[0], b.shape[1]
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, m))
    big = np.zeros((n + m, n + m), dtype=np.result_type(a, b, float))
    big[:n, :n] = a
    big[:n, n:] = b
    e = expm(big * dt)
    return e[:n, :n], e[:n, n:]


def foh_matrices(a, b, dt: float):
    """Phi, Gamma1 = int exp(As) ds B and Gamma2 = (1/dt) int_0^dt exp(A(dt-s)) s ds B."""
    n, m = a.shape[0], b.shape[1]
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, m)), np.zeros((0, m))
    big = np.zeros((n + 2 * m, n + 2 * m), dtype=np.result_type(a, b, float))
    big[:n, :n] = a
    big[:n, n:n + m] = b
    big[n:n + m, n + m:] = np.eye(m) / dt
    e = expm(big * dt)
    return e[:n, :n], e[:n, n:n + m], e[:n, n + m:]


# -- gramians and norms -----------------------------------------------------------------------

def gram(a, b, discrete: bool) -> np.ndarray:
    q = b @ b.conj().T
    return sym(lyap_d(a, q) if discrete else lyap_c(a, q))


def is_stable(poles, discrete: bool) -> bool:
    poles = np.asarray(poles)
    if poles.size == 0:
        return True
    if discrete:
        return bool(np.all(np.abs(poles) < 1))
    return bool(np.all(poles.real < 0))


def h2_norm(a, b, c, d, discrete: bool) -> float:
    if a.shape[0] == 0:
        if discrete:
            return float(np.sqrt(np.sum(np.abs(d) ** 2)))
        return 0.0 if not np.any(d != 0) else np.inf
    if not is_stable(eigvals(a), discrete):
        return np.inf
    if not discrete and np.any(d != 0):
        return np.inf
    wc = gram(a, b, discrete)
    val = np.trace(c @ wc @ c.conj().T).real
    if discrete:
        val += np.sum(np.abs(d) ** 2)
    return float(np.sqrt(max(val, 0.0)))


def sigma_max(a, b, c, d, s) -> float:
    h = ss_freqresp(a, b, c, d, [s])[:, :, 0]
    if not np.all(np.isfinite(h)):
        return np.inf
    return float(np.linalg.norm(h, 2)) if h.size else 0.0


def hinf_norm(a, b, c, d, ts: float, tol: float = 1e-9):
    """Peak gain (largest singular value over frequency) and the frequency where it occurs."""
    discrete = ts != 0
    n = a.shape[0]
    if d.size == 0:
        return 0.0, 0.0
    if n == 0:
        return float(np.linalg.norm(d, 2)), 0.0
    poles = eigvals(a)
    tsv = abs(ts) if ts > 0 else 1.0

    def point(w):
        return np.exp(1j * w * tsv) if discrete else 1j * w

    # a pole on the stability boundary gives an infinite peak
    if discrete:
        on = np.abs(np.abs(poles) - 1) < 1e-10
        if np.any(on):
            w0 = float(np.min(np.abs(np.angle(poles[on])))) / tsv
            return np.inf, w0
        wmax = np.pi / tsv
        mags = np.abs(np.log(poles[np.abs(poles) > 0].astype(complex))) / tsv
    else:
        on = np.abs(poles.real) < 1e-10 * np.maximum(np.abs(poles), 1.0)
        if np.any(on):
            return np.inf, float(np.min(np.abs(poles[on].imag)))
        wmax = None
        mags = np.abs(poles)
    mags = mags[mags > 0]
    lo = (mags.min() if mags.size else 1.0) * 1e-3
    hi = (mags.max() if mags.size else 1.0) * 1e3
    if discrete:
        hi = wmax
        lo = min(lo, wmax * 1e-4)
    grid = np.concatenate([[0.0], np.logspace(np.log10(lo), np.log10(hi), 400)])
    if discrete:
        grid = np.concatenate([grid, [wmax]])
    else:
        grid = np.concatenate([grid, [np.inf]])
    extra = []
    for p in poles:  # resonances sit near the pole frequencies
        if discrete:
            w = abs(np.angle(p)) / tsv
        else:
            w = abs(p.imag)
        if w > 0:
            extra.extend([w * f for f in (0.98, 0.995, 1.0, 1.005, 1.02)])
    grid = np.unique(np.concatenate([grid, np.array(extra)])) if extra else grid
    if discrete:
        grid = grid[grid <= wmax]
    vals = np.array([sigma_max(a, b, c, d, point(w)) if np.isfinite(w) else float(np.linalg.norm(d, 2))
                     for w in grid])
    k = int(np.argmax(vals))
    best, wbest = vals[k], grid[k]
    if not np.isfinite(wbest):
        fin = vals[:-1]
        if fin.size and np.max(fin) >= best * (1 - 1e-12):
            k = int(np.argmax(fin))
            best, wbest = fin[k], grid[k]
        else:
            return float(best), np.inf
    # polish with a bounded scalar search between the neighbouring grid points
    lo_w = grid[k - 1] if k > 0 else grid[k]
    hi_w = grid[k + 1] if k + 1 < len(grid) and np.isfinite(grid[k + 1]) else grid[k]
    if hi_w > lo_w:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda w: -sigma_max(a, b, c, d, point(w)), bounds=(lo_w, hi_w),
                              method="bounded", options={"xatol": max(tol * max(wbest, 1e-12), 1e-15)})
        if -res.fun > best:
            best, wbest = -res.fun, float(res.x)
    # the response at zero frequency is exact there; prefer it when it ties
    v0 = vals[0]
    if v0 >= best * (1 - 1e-12):
        best, wbest = v0, 0.0
    return float(best), float(wbest)


# -- delays -----------------------------------------------------------------------------------

def pade_poly(tau: float, n: int):
    """Numerator and denominator of the (n, n) Pade approximant of exp(-tau*s); the denominator is monic."""
    if tau == 0 or n == 0:
        return np.ones(1), np.ones(1)
    c = [1.0]
    for k in range(1, n + 1):
        c.append(c[-1] * (n - k + 1) / (k * (2 * n - k + 1)))
    asc = np.array(c) * tau ** np.arange(n + 1)
    den = asc[::-1]
    num = (asc * (-1.0) ** np.arange(n + 1))[::-1]
    return num / den[0], den / den[0]


# -- frequency-domain polynomial tools --------------------------------------------------------

def jw_parts(p):
    """Real and imaginary parts of p(jw) as polynomials in w (real coefficients assumed)."""
    p = ptrim(np.real(p))
    n = p.size - 1
    re, im = np.zeros(n + 1), np.zeros(n + 1)
    for idx, c in enumerate(p):
        k = (n - idx) % 4
        if k == 0:
            re[idx] = c
        elif k == 1:
            im[idx] = c
        elif k == 2:
            re[idx] = -c
        else:
            im[idx] = -c
    return re, im


def even_part(p) -> np.ndarray:
    """q with q(w^2) equal to the even-power part of p(w)."""
    p = np.atleast_1d(p)
    return ptrim(p[::-1][0::2][::-1])


def odd_part(p) -> np.ndarray:
    """q with w*q(w^2) equal to the odd-power part of p(w)."""
    p = np.atleast_1d(p)
    if p.size < 2:
        return np.zeros(1)
    return ptrim(p[::-1][1::2][::-1])


def abs2_x(p) -> np.ndarray:
    """|p(jw)|^2 as a polynomial in x = w^2."""
    re, im = jw_parts(p)
    full = np.polyadd(np.convolve(re, re), np.convolve(im, im))
    return even_part(full)


def positive_roots(px, tol: float = 1e-6) -> np.ndarray:
    """Real positive roots of a polynomial (double roots that split into a close pair count once)."""
    px = ptrim(px)
    if px.size <= 1:
        return np.zeros(0)
    r = np.roots(px)
    scale = np.maximum(np.abs(r), 1e-300)
    keep = (np.abs(r.imag) <= tol * scale) & (r.real > 0)
    vals = np.sort(r.real[keep])
    out: list = []
    for v in vals:
        if out and abs(v - out[-1]) <= 10 * tol * max(v, out[-1]) and np.any(np.abs(r.imag[keep]) > 0):
            out[-1] = (out[-1] + v) / 2
        else:
            out.append(float(v))
    return np.array(out)


def wplane(num, den):
    """The polynomials n((1+v)/(1-v)) (1-v)^N and d(...) (1-v)^N: z on the unit circle is v = j*tan(w*Ts/2)."""
    num, den = ptrim(num), ptrim(den)
    big = max(num.size, den.size) - 1

    def conv(p):
        out = np.zeros(1)
        deg = p.size - 1
        for idx, c in enumerate(p):
            k = deg - idx
            term = np.array([c], dtype=float)
            for _ in range(k):
                term = np.convolve(term, [1.0, 1.0])
            for _ in range(big - k):
                term = np.convolve(term, [-1.0, 1.0])
            out = np.polyadd(out, term)
        return out

    return conv(np.real(num)), conv(np.real(den))
