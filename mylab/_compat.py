"""Small fallbacks for older NumPy and SciPy releases (and the ones bundled with Pyodide).

mylab supports NumPy >= 1.26 and SciPy >= 1.13. A few SciPy features it uses arrived later; the
helpers here use them when present and compute the same result another way otherwise.
"""

from __future__ import annotations

import numpy as np


def triangular_rcond(a: np.ndarray, lower: bool) -> float:
    """LAPACK's 1-norm reciprocal condition estimate of a triangular matrix (the xTRCON routine).

    SciPy exposes ``trcon`` only from 1.15. Without it, the estimate comes from ``gecon`` applied to
    the matrix as its own LU factorisation: an upper triangular U is the LU factor of itself with a
    unit L. A lower triangular L is handled through its transpose, whose infinity norm is the
    1-norm of L.
    """
    from scipy.linalg import lapack
    n = a.shape[0]
    if n == 0:
        return float("inf")
    try:
        trcon = lapack.get_lapack_funcs("trcon", (a,))
    except (ValueError, AttributeError):
        trcon = None
    if trcon is not None:
        return float(trcon(a, norm="1", uplo="L" if lower else "U", diag="N")[0])
    if np.any(np.diag(a) == 0):
        return 0.0
    gecon = lapack.get_lapack_funcs("gecon", (a,))
    if lower:
        u = np.triu(np.asarray(a).T)
        anorm = float(np.abs(u).sum(axis=1).max())  # infinity norm of L', the 1-norm of L
        return float(gecon(u, anorm, norm="I")[0])
    u = np.triu(np.asarray(a))
    anorm = float(np.abs(u).sum(axis=0).max())
    return float(gecon(u, anorm, norm="1")[0])


def makima(x, y, axis: int = 0, extrapolate: bool = True):
    """A modified Akima interpolant (SciPy's ``Akima1DInterpolator`` with ``method='makima'``).

    SciPy accepts ``extrapolate`` in the constructor only from 1.14; on older versions the
    attribute is set afterwards, which has the same effect on evaluation.
    """
    from scipy.interpolate import Akima1DInterpolator
    try:
        return Akima1DInterpolator(x, y, axis=axis, method="makima", extrapolate=extrapolate)
    except TypeError:
        obj = Akima1DInterpolator(x, y, axis=axis, method="makima")
        obj.extrapolate = extrapolate
        return obj
