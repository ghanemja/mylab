"""Multivariate analysis: principal components, factor analysis, canonical correlation,
multidimensional scaling, Procrustes analysis, Mahalanobis distance and t-SNE."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ._registry import builtin
from ._stats_util import as_bool, choice, get_opts, given, intval, is_empty_arg, outs, rng_of, scalar, struct, vec
from ._util import fnum, is_text, text


def _sign_fix(V: np.ndarray) -> np.ndarray:
    """Flip columns so that the entry of largest magnitude in each is positive."""
    if V.size == 0:
        return np.ones(V.shape[1])
    big = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[big, np.arange(V.shape[1])])
    s[s == 0] = 1
    return s


# -- principal components ----------------------------------------------------------------------------------

_PCA_OPTS = ["Algorithm", "Centered", "Economy", "NumComponents", "Rows", "Weights", "VariableWeights", "Coeff0",
             "Score0", "Options"]


@builtin("pca")
def _pca(interp, args, nargout):
    pos, opts = get_opts(args, 1, _PCA_OPTS, "pca")
    if len(pos) != 1:
        raise MatlabError("pca: expected pca(X, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError("pca: the data must be a matrix with one observation per row.")
    algo = choice(opts["algorithm"], ["svd", "eig", "als"], "Algorithm") if "algorithm" in opts else "svd"
    if algo == "als" or "coeff0" in opts or "score0" in opts:
        raise MatlabError("pca: the alternating least squares algorithm ('als') is not supported.")
    rows = choice(opts["rows"], ["complete", "pairwise", "all"], "Rows") if "rows" in opts else "complete"
    centered = as_bool(opts["centered"], "Centered") if "centered" in opts else True
    economy = as_bool(opts["economy"], "Economy") if "economy" in opts else True
    n_all, p = X.shape
    has_nan = np.isnan(X).any()
    if rows == "all" and has_nan:
        raise MatlabError("pca: the data contains NaN. Use 'Rows', 'complete' (the default) to skip those rows.")
    w = np.ones(n_all)
    if "weights" in opts:
        w = vec(opts["weights"])
        if w.size != n_all or np.any(w < 0):
            raise MatlabError("pca: Weights must hold one non-negative value per observation.")
    keep = ~np.isnan(X).any(axis=1) & ~np.isnan(w) if rows != "pairwise" else np.ones(n_all, dtype=bool)
    Xc, wk = X[keep], w[keep]
    n = Xc.shape[0]
    vw = np.ones(p)
    if "variableweights" in opts:
        v = opts["variableweights"]
        if is_text(v):
            if text(v).lower() != "variance":
                raise MatlabError("pca: VariableWeights must be 'variance' or a vector of weights.")
            with np.errstate(all="ignore"):
                vw = 1.0 / np.nanvar(Xc, axis=0, ddof=1)
        else:
            vw = vec(v)
            if vw.size != p or np.any(vw <= 0):
                raise MatlabError("pca: VariableWeights must hold one positive value per variable.")
    if n == 0:
        z = np.zeros((0, 1))
        return (np.zeros((p, 0)), np.zeros((n_all, 0)), z, z, z, np.zeros((1, p)))[:max(nargout, 1)] \
            if nargout > 1 else np.zeros((p, 0))
    wn = wk * (n / wk.sum()) if wk.sum() > 0 else wk
    dof = max(n - (1 if centered else 0), 1)
    if rows == "pairwise":
        if algo != "eig":
            if has_nan and "algorithm" in opts:
                raise MatlabError("pca: 'Rows', 'pairwise' needs 'Algorithm', 'eig'.")
            algo = "eig"
        mu = np.nanmean(X, axis=0, keepdims=True) if centered else np.zeros((1, p))
        Z = (X - mu) * np.sqrt(vw)
        C = np.empty((p, p))
        for i in range(p):
            for j in range(i, p):
                ok = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
                C[i, j] = C[j, i] = np.sum(Z[ok, i] * Z[ok, j]) / max(ok.sum() - (1 if centered else 0), 1)
        evals, V = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        latent_all, V = np.clip(evals[order], 0, None), V[:, order]
        nvalid = p
    else:
        mu = (wn @ Xc / n).reshape(1, -1) if centered else np.zeros((1, p))
        Z = (Xc - mu) * np.sqrt(vw)
        Zw = Z * np.sqrt(wn)[:, None]
        if algo == "eig":
            evals, V = np.linalg.eigh(Zw.T @ Zw / dof)
            order = np.argsort(evals)[::-1]
            latent_all, V = np.clip(evals[order], 0, None), V[:, order]
        else:
            _, s, Vt = np.linalg.svd(Zw, full_matrices=not economy or n < p)
            V = Vt.T
            latent_all = np.zeros(V.shape[1])
            latent_all[:s.size] = s ** 2 / dof
        nvalid = min(dof if centered else n, p) if economy else p
        if V.shape[1] < nvalid:
            nvalid = V.shape[1]
    V = V[:, :nvalid] * _sign_fix(V[:, :nvalid])
    latent_all = latent_all[:nvalid]
    ncomp = nvalid
    if "numcomponents" in opts:
        ncomp = min(ncomp, intval(opts["numcomponents"], "NumComponents"))
    coeff_all = V / np.sqrt(vw)[:, None]
    Zfull = (X - mu) * np.sqrt(vw)
    score_all = Zfull @ V
    if rows != "pairwise":
        score_all[~keep] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        std_scores = score_all / np.sqrt(latent_all)
    std_scores[:, latent_all <= 1e-14 * max(latent_all.max(initial=0.0), 1e-300)] = 0
    t2 = np.sum(std_scores ** 2, axis=1, keepdims=True)
    total = latent_all.sum()
    explained = (100 * latent_all / total if total > 0 else np.zeros_like(latent_all)).reshape(-1, 1)
    res = (coeff_all[:, :ncomp], score_all[:, :ncomp], latent_all.reshape(-1, 1), t2, explained, mu)
    return res[0] if nargout <= 1 else res[:nargout]


@builtin("pcacov")
def _pcacov(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("pcacov: expected pcacov(V).")
    C = np.asarray(fnum(args[0]), dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise MatlabError("pcacov: the input must be a square covariance matrix.")
    U, s, _ = np.linalg.svd((C + C.T) / 2)
    U = U * _sign_fix(U)
    total = s.sum()
    return outs(nargout, U, s.reshape(-1, 1), (100 * s / total if total > 0 else 0 * s).reshape(-1, 1))


@builtin("pcares")
def _pcares(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("pcares: expected pcares(X, ndim).")
    X = np.asarray(fnum(args[0]), dtype=float)
    k = intval(args[1], "ndim")
    if not 0 <= k <= X.shape[1]:
        raise MatlabError("pcares: ndim must be between 0 and the number of columns of X.")
    mu = np.nanmean(X, axis=0)
    Z = X - mu
    ok = ~np.isnan(Z).any(axis=1)
    _, _, Vt = np.linalg.svd(Z[ok], full_matrices=False)
    V = Vt[:k].T
    recon = Z @ V @ V.T + mu
    return outs(nargout, X - recon, recon)


# -- factor analysis ---------------------------------------------------------------------------------------------

def _rotate(L: np.ndarray, kind: str, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal varimax/quartimax rotation; returns the rotated loadings and the rotation matrix."""
    d, m = L.shape
    if m < 2 or kind == "none":
        return L, np.eye(m)
    gamma = 1.0 if kind == "varimax" else 0.0
    h = np.sqrt(np.sum(L ** 2, axis=1)) if normalize else np.ones(d)
    h = np.where(h > 0, h, 1.0)
    A = L / h[:, None]
    T = np.eye(m)
    last = 0.0
    for _ in range(1000):
        B = A @ T
        U, s, Vt = np.linalg.svd(A.T @ (B ** 3 - (gamma / d) * B @ np.diag(np.sum(B ** 2, axis=0))))
        T = U @ Vt
        cur = s.sum()
        if cur - last < 1e-12 * max(cur, 1.0):
            break
        last = cur
    return (A @ T) * h[:, None], T


@builtin("factoran")
def _factoran(interp, args, nargout):
    import scipy.stats as st
    from scipy.optimize import minimize
    if len(args) < 2:
        raise MatlabError("factoran: expected factoran(X, m, 'Name', value, ...).")
    known = ["xtype", "scores", "start", "rotate", "coeff", "normalize", "nobs", "delta", "optimopts", "maxit",
             "reltol", "power", "userargs"]
    pos, opts = get_opts(args, 2, known, "factoran")
    if len(pos) != 2:
        raise MatlabError("factoran: expected factoran(X, m, 'Name', value, ...).")
    for o in ("userargs", "coeff", "power"):
        if o in opts:
            raise MatlabError(f"factoran: the option '{o}' is not supported.")
    X = np.asarray(fnum(pos[0]), dtype=float)
    m = intval(pos[1], "m")
    xtype = choice(opts["xtype"], ["data", "covariance"], "xtype") if "xtype" in opts else "data"
    rot = choice(opts["rotate"], ["none", "varimax", "quartimax", "equamax", "parsimax", "orthomax", "promax",
                                  "procrustes", "pattern"], "rotate") if "rotate" in opts else "varimax"
    if rot not in ("none", "varimax", "quartimax"):
        raise MatlabError(f"factoran: the rotation '{rot}' is not supported (use 'none', 'varimax' or 'quartimax').")
    normalize = as_bool(opts["normalize"], "normalize") if "normalize" in opts else True
    delta = scalar(opts["delta"], "delta") if "delta" in opts else 0.005
    if xtype == "data":
        X = X[~np.isnan(X).any(axis=1)]
        n, d = X.shape
        mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
        Zs = (X - mu) / sd
        R = np.corrcoef(X, rowvar=False)
    else:
        d = X.shape[0]
        sdv = np.sqrt(np.diag(X))
        R = X / np.outer(sdv, sdv)
        n = intval(opts["nobs"], "nobs") if "nobs" in opts else None
        Zs = None
    dfe = ((d - m) ** 2 - (d + m)) // 2
    if m < 1 or dfe < 0:
        raise MatlabError("factoran: there are too many factors for the number of variables.")

    def parts(psi):
        sp = np.sqrt(psi)
        ev, V = np.linalg.eigh(R / np.outer(sp, sp))
        order = np.argsort(ev)[::-1]
        return ev[order], V[:, order], sp

    def obj(lp):
        psi = np.exp(lp)
        ev, _, _ = parts(psi)
        tail = np.clip(ev[m:], 1e-300, None)
        return float(np.sum(tail - np.log(tail) - 1))

    if "start" in opts and not is_text(opts["start"]):
        start = vec(opts["start"])
        if start.size != d:
            raise MatlabError("factoran: start must hold one specific variance per variable.")
    else:
        if "start" in opts and text(opts["start"]).lower() not in ("rsquared", "random"):
            raise MatlabError("factoran: start must be 'Rsquared', 'random' or a vector of specific variances.")
        with np.errstate(all="ignore"):
            start = (1 - 0.5 * m / d) / np.diag(np.linalg.pinv(R))
        if "start" in opts and text(opts["start"]).lower() == "random":
            start = rng_of(interp).random(d)
    start = np.clip(start, delta, 1.0)
    sol = minimize(obj, np.log(start), method="L-BFGS-B", bounds=[(np.log(delta), 0.0)] * d,
                   options={"maxiter": intval(opts["maxit"], "maxit") if "maxit" in opts else 2000, "ftol": 1e-14,
                            "gtol": 1e-10})
    psi = np.exp(sol.x)
    ev, V, sp = parts(psi)
    L = sp[:, None] * V[:, :m] * np.sqrt(np.clip(ev[:m] - 1, 0, None))
    L = L * np.where(L.sum(axis=0) < 0, -1.0, 1.0)
    L, T = _rotate(L, rot, normalize)
    flip = np.where(L.sum(axis=0) < 0, -1.0, 1.0)
    L, T = L * flip, T * flip
    f = sol.fun
    stats: dict[str, Any] = {"loglike": float(-0.5 * (f + np.log(max(np.linalg.det(R), 1e-300)) + d)),
            "dfe": float(dfe)}
    if n is not None:
        stats["loglike"] = float(stats["loglike"] * n)
        if dfe > 0:
            chisq = (n - 1 - (2 * d + 5) / 6 - 2 * m / 3) * f
            stats["chisq"] = float(chisq)
            stats["p"] = float(st.chi2.sf(chisq, dfe))
    F = np.zeros((0, 0))
    if nargout > 4:
        if Zs is None:
            raise MatlabError("factoran: factor scores need the raw data, not a covariance matrix.")
        kind = choice(opts["scores"], ["wls", "Bartlett", "regression", "Thomson"], "scores") if "scores" in opts \
            else "wls"
        if kind in ("wls", "bartlett"):
            A = L / psi[:, None]
            F = Zs @ A @ np.linalg.inv(L.T @ A)
        else:
            F = Zs @ np.linalg.solve(L @ L.T + np.diag(psi), L)
    return outs(nargout, L, psi.reshape(-1, 1), T, struct(**stats), F)


# -- canonical correlation --------------------------------------------------------------------------------------

@builtin("canoncorr")
def _canoncorr(interp, args, nargout):
    import scipy.linalg as sla
    import scipy.stats as st
    if len(args) != 2:
        raise MatlabError("canoncorr: expected canoncorr(X, Y).")
    X = np.asarray(fnum(args[0]), dtype=float)
    Y = np.asarray(fnum(args[1]), dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise MatlabError("canoncorr: X and Y must have the same number of rows.")
    n, p1 = X.shape
    p2 = Y.shape[1]
    if n < 2:
        raise MatlabError("canoncorr: at least two observations are needed.")
    X0, Y0 = X - X.mean(axis=0), Y - Y.mean(axis=0)
    Q1, T11, perm1 = sla.qr(X0, mode="economic", pivoting=True)
    Q2, T22, perm2 = sla.qr(Y0, mode="economic", pivoting=True)
    tol1 = np.finfo(float).eps * abs(T11[0, 0]) * max(n, p1)
    tol2 = np.finfo(float).eps * abs(T22[0, 0]) * max(n, p2)
    r1 = int(np.sum(np.abs(np.diag(T11)) > tol1))
    r2 = int(np.sum(np.abs(np.diag(T22)) > tol2))
    if r1 == 0 or r2 == 0:
        raise MatlabError("canoncorr: X and Y must not be constant.")
    Q1, T11 = Q1[:, :r1], T11[:r1, :r1]
    Q2, T22 = Q2[:, :r2], T22[:r2, :r2]
    d = min(r1, r2)
    L, D, Mt = np.linalg.svd(Q1.T @ Q2)
    M = Mt.T
    A = np.zeros((p1, d))
    B = np.zeros((p2, d))
    A[perm1[:r1]] = np.linalg.solve(T11, L[:, :d]) * np.sqrt(n - 1)
    B[perm2[:r2]] = np.linalg.solve(T22, M[:, :d]) * np.sqrt(n - 1)
    r = np.clip(D[:d], 0, 1)
    U, V = X0 @ A, Y0 @ B
    if nargout < 6:
        return outs(nargout, A, B, r.reshape(1, -1), U, V)
    k = np.arange(d)
    with np.errstate(all="ignore"):
        wilks = np.array([np.prod(1 - r[i:] ** 2) for i in range(d)])
        pk, qk = r1 - k, r2 - k
        df1 = (pk * qk).astype(float)
        chisq = -(n - 1 - (r1 + r2 + 1) / 2.0) * np.log(wilks)
        s = np.where(pk ** 2 + qk ** 2 > 5, np.sqrt((pk ** 2 * qk ** 2 - 4.0) / (pk ** 2 + qk ** 2 - 5.0)), 1.0)
        df2 = (n - 1 - (r1 + r2 + 1) / 2.0) * s - df1 / 2 + 1
        lam = wilks ** (1 / s)
        F = (1 - lam) / lam * df2 / df1
        pF = st.f.sf(F, df1, df2)
        pChi = st.chi2.sf(chisq, df1)
    stats = struct(Wilks=wilks.reshape(1, -1), df1=df1.reshape(1, -1), df2=df2.reshape(1, -1), F=F.reshape(1, -1),
                   pF=pF.reshape(1, -1), chisq=chisq.reshape(1, -1), pChisq=pChi.reshape(1, -1),
                   dfe=df1.reshape(1, -1), p=pChi.reshape(1, -1))
    return A, B, r.reshape(1, -1), U, V, stats


# -- multidimensional scaling ----------------------------------------------------------------------------------------

def _square_dissim(D_arg: Any, fname: str) -> tuple[np.ndarray, bool]:
    """Full symmetric dissimilarity matrix from a square matrix or a pdist-style vector."""
    D = np.asarray(fnum(D_arg), dtype=float)
    if D.ndim == 2 and D.shape[0] == D.shape[1] and D.shape[0] > 1:
        if np.allclose(np.diag(D), 1) and np.all(D <= 1 + 1e-12):
            D = np.sqrt(np.clip(1 - D, 0, None))       # a similarity matrix
        if not np.allclose(D, D.T, atol=1e-8 * max(1.0, np.nanmax(np.abs(D)))):
            raise MatlabError(f"{fname}: the dissimilarity matrix must be symmetric.")
        return (D + D.T) / 2, False
    v = D.flatten()
    n = int(round((1 + np.sqrt(1 + 8 * v.size)) / 2))
    if n * (n - 1) // 2 != v.size or v.size == 0:
        raise MatlabError(f"{fname}: expected a square dissimilarity matrix or a vector as returned by pdist.")
    M = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    M[iu] = v
    return M + M.T, True


def _cmds(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ (D ** 2) @ J
    ev, V = np.linalg.eigh((Bm + Bm.T) / 2)
    order = np.argsort(ev)[::-1]
    return ev[order], V[:, order]


@builtin("cmdscale")
def _cmdscale(interp, args, nargout):
    if not 1 <= len(args) <= 2:
        raise MatlabError("cmdscale: expected cmdscale(D) or cmdscale(D, p).")
    D, _ = _square_dissim(args[0], "cmdscale")
    ev, V = _cmds(D)
    keep = ev > max(abs(ev[0]), 1e-300) * np.finfo(float).eps ** 0.75
    k = int(keep.sum())
    if given(args, 1):
        k = min(k, intval(args[1], "p"))
    Y = V[:, :k] * np.sqrt(ev[:k])
    Y = Y * _sign_fix(Y)
    return outs(nargout, Y, ev.reshape(-1, 1))


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression by pooling adjacent violators."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw if tw > 0 else (vals[-2] + vals[-1]) / 2
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [tw], [c]
    return np.repeat(vals, cnt)


def _pair_dist(Y: np.ndarray, iu) -> np.ndarray:
    diff = Y[iu[0]] - Y[iu[1]]
    return np.sqrt(np.sum(diff ** 2, axis=1))


@builtin("mdscale")
def _mdscale(interp, args, nargout):
    from scipy.optimize import minimize
    if len(args) < 2:
        raise MatlabError("mdscale: expected mdscale(D, p, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["Criterion", "Weights", "Start", "Replicates", "Options"], "mdscale")
    if len(pos) != 2:
        raise MatlabError("mdscale: expected mdscale(D, p, 'Name', value, ...).")
    D, was_vec = _square_dissim(pos[0], "mdscale")
    p = intval(pos[1], "p")
    n = D.shape[0]
    crit = choice(opts["criterion"], ["stress", "sstress", "metricstress", "metricsstress", "sammon", "strain"],
                  "Criterion") if "criterion" in opts else "stress"
    iu = np.triu_indices(n, 1)
    delta = D[iu]
    if np.any(delta < 0) or np.isnan(delta).any():
        raise MatlabError("mdscale: dissimilarities must be non-negative and must not be NaN.")
    w = np.ones(delta.size)
    if "weights" in opts:
        W, _ = _square_dissim(opts["weights"], "mdscale")
        w = W[iu]
    reps = intval(opts["replicates"], "Replicates") if "replicates" in opts else 1
    rng = rng_of(interp)

    def classical():
        ev, V = _cmds(D)
        k = min(p, int(np.sum(ev > 1e-12 * max(ev[0], 1e-300))))
        Y0 = np.zeros((n, p))
        Y0[:, :k] = V[:, :k] * np.sqrt(np.clip(ev[:k], 0, None))
        if k < p:
            Y0[:, k:] = 1e-3 * np.std(Y0[:, :max(k, 1)]) * rng.standard_normal((n, p - k))
        return Y0

    if crit == "strain":
        Y = classical()
        d = _pair_dist(Y, iu)
        stress = float(np.sqrt(np.sum((d ** 2 - delta ** 2) ** 2) / np.sum(delta ** 4))) if np.any(delta > 0) else 0.0
        return _mds_out(nargout, Y * _sign_fix(Y), stress, delta, iu, n, was_vec)

    start = opts.get("start")
    starts = []
    if start is not None and not is_text(start):
        Y0 = np.asarray(fnum(start), dtype=float)
        if Y0.shape[:2] != (n, p):
            raise MatlabError(f"mdscale: Start must be a {n}-by-{p} matrix.")
        starts = [Y0[:, :, k] for k in range(Y0.shape[2])] if Y0.ndim == 3 else [Y0]
    else:
        mode = choice(start, ["cmdscale", "random"], "Start") if start is not None else "cmdscale"
        for _ in range(reps):
            starts.append(classical() if mode == "cmdscale" else rng.standard_normal((n, p)) * (delta.std() or 1.0))
    nonmetric = crit in ("stress", "sstress")
    squared = crit in ("sstress", "metricsstress")
    order = np.argsort(delta, kind="mergesort")

    def disparities(d):
        if not nonmetric:
            return delta
        t = np.empty_like(d)
        src = d ** 2 if squared else d
        t[order] = _pava(src[order], w[order])
        return np.sqrt(np.clip(t, 0, None)) if squared else t

    def loss_grad(yflat, target):
        Y = yflat.reshape(n, p)
        diff = Y[iu[0]] - Y[iu[1]]
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        ds = np.where(d > 0, d, 1e-12)
        if squared:
            e = d ** 2 - target ** 2
            f = np.sum(w * e ** 2)
            coef = 4 * w * e
        elif crit == "sammon":
            t = np.where(target > 0, target, 1e-12)
            e = d - target
            f = np.sum(w * e ** 2 / t)
            coef = 2 * w * e / (t * ds)
        else:
            e = d - target
            f = np.sum(w * e ** 2)
            coef = 2 * w * e / ds
        G = np.zeros((n, p))
        np.add.at(G, iu[0], coef[:, None] * diff)
        np.add.at(G, iu[1], -coef[:, None] * diff)
        return float(f), G.ravel()

    def final_stress(d, target):
        with np.errstate(all="ignore"):
            if squared:
                return float(np.sqrt(np.sum(w * (d ** 2 - target ** 2) ** 2) / np.sum(w * d ** 4)))
            if crit == "sammon":
                t = np.where(target > 0, target, 1e-12)
                return float(np.sum(w * (d - target) ** 2 / t) / np.sum(w * target))
            return float(np.sqrt(np.sum(w * (d - target) ** 2) / np.sum(w * d ** 2)))

    best = None
    for Y0 in starts:
        y = Y0.ravel().copy()
        target = disparities(_pair_dist(Y0, iu))
        prev = np.inf
        for _ in range(200 if nonmetric else 1):
            sol = minimize(loss_grad, y, args=(target,), jac=True, method="L-BFGS-B",
                           options={"maxiter": 500 if not nonmetric else 60, "ftol": 1e-15, "gtol": 1e-10})
            y = sol.x
            d = _pair_dist(y.reshape(n, p), iu)
            if not nonmetric:
                break
            if np.sum(d ** 2) > 0:              # keep the configuration at a fixed scale
                y = y / np.sqrt(np.sum(d ** 2) / d.size)
                d = _pair_dist(y.reshape(n, p), iu)
            target = disparities(d)
            cur = final_stress(d, target)
            if abs(prev - cur) < 1e-10:
                break
            prev = cur
        d = _pair_dist(y.reshape(n, p), iu)
        target = disparities(d)
        s = final_stress(d, target)
        if best is None or s < best[0]:
            best = (s, y.reshape(n, p).copy(), target)
    stress, Y, target = best
    Y = Y - Y.mean(axis=0)
    return _mds_out(nargout, Y, stress, target, iu, n, was_vec)


def _mds_out(nargout, Y, stress, target, iu, n, was_vec):
    if nargout <= 1:
        return Y
    if was_vec:
        disp = target.reshape(1, -1)
    else:
        disp = np.zeros((n, n))
        disp[iu] = target
        disp = disp + disp.T
    return outs(nargout, Y, stress, disp)


@builtin("procrustes")
def _procrustes(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("procrustes: expected procrustes(X, Y, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["scaling", "reflection"], "procrustes")
    if len(pos) != 2:
        raise MatlabError("procrustes: expected procrustes(X, Y, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    Y = np.asarray(fnum(pos[1]), dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise MatlabError("procrustes: X and Y must have the same number of rows.")
    if Y.shape[1] > X.shape[1]:
        raise MatlabError("procrustes: Y must not have more columns than X.")
    scaling = as_bool(opts["scaling"], "scaling") if "scaling" in opts else True
    refl: Any = "best"
    if "reflection" in opts:
        v = opts["reflection"]
        refl = "best" if (is_text(v) and text(v).lower() == "best") else as_bool(v, "reflection")
    n, m = X.shape
    my = Y.shape[1]
    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - muX, Y - muY
    ssX, ssY = np.sum(X0 ** 2), np.sum(Y0 ** 2)
    if ssX == 0:
        raise MatlabError("procrustes: X must not be a single repeated point.")
    normX, normY = np.sqrt(ssX), np.sqrt(ssY)
    if ssY == 0:
        Z = np.tile(muX, (n, 1))
        tr = struct(T=np.eye(my, m), b=1.0, c=np.tile(muX, (n, 1)))
        return outs(nargout, 1.0, Z, tr)
    Xn, Yn = X0 / normX, Y0 / normY
    if my < m:
        Yn = np.hstack([Yn, np.zeros((n, m - my))])
    U, s, Vt = np.linalg.svd(Xn.T @ Yn)
    V = Vt.T
    T = V @ U.T
    if refl != "best":
        have = np.linalg.det(T) < 0
        if have != refl:
            V[:, -1] = -V[:, -1]
            s = s.copy()
            s[-1] = -s[-1]
            T = V @ U.T
    trace = s.sum()
    if scaling:
        b = trace * normX / normY
        d = 1 - trace ** 2
        Z = normX * trace * Yn @ T + muX
    else:
        b = 1.0
        d = 1 + ssY / ssX - 2 * trace * normY / normX
        Z = normY * Yn @ T + muX
    if my < m:
        T = T[:my, :]
    c = muX - b * muY @ T
    tr = struct(T=T, b=float(b), c=np.tile(c, (n, 1)))
    return outs(nargout, float(d), Z, tr)


@builtin("mahal")
def _mahal(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("mahal: expected mahal(Y, X).")
    Y = np.asarray(fnum(args[0]), dtype=float)
    X = np.asarray(fnum(args[1]), dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise MatlabError("mahal: Y and X must have the same number of columns.")
    n, p = X.shape
    if n <= p:
        raise MatlabError("mahal: X must have more rows than columns.")
    mu = X.mean(axis=0)
    _, R = np.linalg.qr(X - mu)
    try:
        ri = np.linalg.solve(R.T, (Y - mu).T)
    except np.linalg.LinAlgError:
        raise MatlabError("mahal: the covariance of X is singular.") from None
    return (np.sum(ri ** 2, axis=0) * (n - 1)).reshape(-1, 1)


# -- t-SNE ---------------------------------------------------------------------------------------------------

def _perplexity_probs(D2: np.ndarray, perplexity: float) -> np.ndarray:
    """Conditional neighbour probabilities with a per-point bandwidth matching the perplexity."""
    n = D2.shape[0]
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        d = np.delete(D2[i], i)
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(100):
            e = np.exp(-(d - d.min()) * beta)
            s = e.sum()
            pr = e / s
            H = -np.sum(pr * np.log(np.maximum(pr, 1e-300)))
            if abs(H - target) < 1e-7:
                break
            if H > target:
                lo = beta
                beta = beta * 2 if np.isinf(hi) else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        P[i, np.arange(n) != i] = pr
    return P


@builtin("tsne")
def _tsne(interp, args, nargout):
    known = ["Algorithm", "Distance", "Exaggeration", "NumDimensions", "NumPCAComponents", "Perplexity",
             "Standardize", "InitialY", "LearnRate", "NumPrint", "Options", "Theta", "Verbose", "BarnesHutTheta",
             "CacheSize"]
    pos, opts = get_opts(args, 1, known, "tsne")
    if len(pos) != 1:
        raise MatlabError("tsne: expected tsne(X, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    X = X[~np.isnan(X).any(axis=1)] if np.isnan(X).any() else X
    n = X.shape[0]
    if "algorithm" in opts:
        choice(opts["algorithm"], ["exact", "barneshut"], "Algorithm")   # both use the exact gradient here
    dims = intval(opts["numdimensions"], "NumDimensions") if "numdimensions" in opts else 2
    perp = scalar(opts["perplexity"], "Perplexity") if "perplexity" in opts else 30.0
    if perp >= n:
        perp = max((n - 1) / 3.0, 1.0)
    exag = scalar(opts["exaggeration"], "Exaggeration") if "exaggeration" in opts else 4.0
    lr = scalar(opts["learnrate"], "LearnRate") if "learnrate" in opts else 500.0
    if "standardize" in opts and as_bool(opts["standardize"], "Standardize"):
        sd = X.std(axis=0, ddof=1)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    npca = intval(opts["numpcacomponents"], "NumPCAComponents") if "numpcacomponents" in opts else 0
    if npca > 0:
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ Vt[:npca].T
    metric = text(opts["distance"]).lower() if "distance" in opts and is_text(opts["distance"]) else "euclidean"
    if "distance" in opts and not is_text(opts["distance"]):
        raise MatlabError("tsne: function-handle distances are not supported.")
    from .stats_cluster import dist_matrix
    D = dist_matrix(interp, X, X, metric, [], "tsne")
    P = _perplexity_probs(D ** 2, perp)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-300)
    max_iter = 1000
    o = opts.get("options")
    if o is not None and not is_empty_arg(o):
        from ..values import Struct
        if not isinstance(o, Struct):
            raise MatlabError("tsne: Options must be a structure, e.g. statset('MaxIter', 500).")
        if "MaxIter" in o.fields and not is_empty_arg(o.get_field(0, "MaxIter")):
            max_iter = intval(o.get_field(0, "MaxIter"), "MaxIter")
    rng = rng_of(interp)
    if "initialy" in opts:
        Y = np.asarray(fnum(opts["initialy"]), dtype=float).copy()
        if Y.shape != (n, dims):
            raise MatlabError(f"tsne: InitialY must be a {n}-by-{dims} matrix.")
    else:
        Y = 1e-4 * rng.standard_normal((n, dims))
    vel = np.zeros_like(Y)
    gains = np.ones_like(Y)
    loss = np.nan
    for it in range(max_iter):
        Pe = P * exag if it < 99 else P
        sq = np.sum(Y ** 2, axis=1)
        num = 1.0 / (1.0 + sq[:, None] + sq[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-300)
        L = (Pe - Q) * num
        grad = 4 * (np.diag(L.sum(axis=1)) - L) @ Y
        gains = np.where(np.sign(grad) != np.sign(vel), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        momentum = 0.5 if it < 250 else 0.8
        vel = momentum * vel - lr * gains * grad
        Y = Y + vel
        Y = Y - Y.mean(axis=0)
    sq = np.sum(Y ** 2, axis=1)
    num = 1.0 / (1.0 + sq[:, None] + sq[None, :] - 2 * Y @ Y.T)
    np.fill_diagonal(num, 0.0)
    Q = np.maximum(num / num.sum(), 1e-300)
    mask = ~np.eye(n, dtype=bool)
    loss = float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))
    return outs(nargout, Y, loss)
