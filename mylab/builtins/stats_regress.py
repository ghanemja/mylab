"""More regression: robust and nonlinear fits, penalised regression (lasso, ridge, elastic net),
partial least squares, multinomial models and polynomial confidence bounds."""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..values import FunctionHandle, Struct, char
from ._registry import builtin
from ._stats_util import (
    Labels, StatObject, as_bool, call, cellstr, choice, get_opts, given, intval, is_categorical, is_empty_arg,
    is_table, make_table, num_jacobian, outs, rng_of, scalar, struct, text_list, vec,
)
from ._util import fnum, is_text, text
from .stats_models import GLMFit, Link, LSFit, _coef_lines, _coef_table, _dist_name, _fmt_g, _glm_xy, _row_mask

# -- robust regression ------------------------------------------------------------------------------------

_WFUN = {
    "andrews": (1.339, lambda r: np.where(np.abs(r) < np.pi, np.where(r == 0, 1.0, np.sin(r) / np.where(r == 0, 1,
            r)), 0.0)),
    "bisquare": (4.685, lambda r: np.where(np.abs(r) < 1, (1 - r ** 2) ** 2, 0.0)),
    "cauchy": (2.385, lambda r: 1 / (1 + r ** 2)),
    "fair": (1.400, lambda r: 1 / (1 + np.abs(r))),
    "huber": (1.345, lambda r: 1 / np.maximum(1, np.abs(r))),
    "logistic": (1.205, lambda r: np.where(r == 0, 1.0, np.tanh(r) / np.where(r == 0, 1, r))),
    "ols": (1.0, lambda r: np.ones_like(r)),
    "talwar": (2.795, lambda r: (np.abs(r) < 1).astype(float)),
    "welsch": (2.985, lambda r: np.exp(-r ** 2)),
}


def _mad_sigma(r: np.ndarray, p: int) -> float:
    rs = np.sort(np.abs(r))
    return float(np.median(rs[max(p, 1) - 1:]) / 0.6745)


def robust_irls(X: np.ndarray, y: np.ndarray, wfun: str, tune: float | None, prior: np.ndarray | None = None,
                max_iter: int = 100):
    """Iteratively reweighted least squares. Returns (beta, weights, info)."""
    wfun = wfun.lower()
    if wfun not in _WFUN:
        raise MatlabError(f"Unknown robust weight function '{wfun}'. Known: " + ", ".join(sorted(_WFUN)) + ".")
    default_tune, fn = _WFUN[wfun]
    tune = default_tune if tune is None else float(tune)
    n, p = X.shape
    pw = np.ones(n) if prior is None else prior
    ols = LSFit(X, y, pw, True)
    b = ols.beta.copy()
    h = np.minimum(ols.lev, 0.9999)
    adj = 1 / np.sqrt(1 - h)
    tiny = 1e-6 * (np.std(y) or 1.0)
    w = np.ones(n)
    for _ in range(max_iter):
        radj = (y - X @ b) * adj * np.sqrt(pw)
        s = max(_mad_sigma(radj, p), tiny)
        w = fn(radj / (s * tune))
        sw = np.sqrt(w * pw)
        new, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        done = np.all(np.abs(new - b) <= np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(new), np.abs(b)))
        b = new
        if done:
            break
    resid = y - X @ b
    radj = resid * adj * np.sqrt(pw)
    mad_s = max(_mad_sigma(radj, p), tiny)
    # asymptotic scale estimate from the psi function (psi(u) = u w(u)) and its derivative
    st_ = mad_s * tune
    u = radj / st_
    delta = 1e-4
    psi = u * fn(u)
    dpsi = ((u + delta) * fn(u + delta) - (u - delta) * fn(u - delta)) / (2 * delta)
    m1 = float(np.mean(dpsi))
    m2 = float(np.mean((dpsi - m1) ** 2))
    with np.errstate(all="ignore"):
        K = 1 + (p / n) * (m2 / m1 ** 2)
        robust_s = K * np.sqrt(np.sum(psi ** 2 * st_ ** 2 * (1 - h)) / (n - p)) / abs(m1) if n > p else np.nan
    ols_s = float(np.sqrt(ols.mse)) if ols.dfe > 0 else np.nan
    info = {"ols_s": ols_s, "robust_s": float(robust_s), "mad_s": mad_s, "h": h, "adj": adj, "resid": resid,
            "xtxi": ols.xtxi, "dfe": n - p, "tune": tune}
    return b, w, info


@builtin("robustfit")
def _robustfit(interp, args, nargout):
    import scipy.stats as st
    if not 2 <= len(args) <= 5:
        raise MatlabError("robustfit: expected robustfit(X, y, wfun, tune, const).")
    X = np.asarray(fnum(args[0]), dtype=float)
    y = vec(args[1])
    if X.ndim == 2 and X.shape[0] == 1 and y.size == X.shape[1]:
        X = X.T
    if X.shape[0] != y.size:
        raise MatlabError("robustfit: X and y must have the same number of rows.")
    wfun = "bisquare"
    if given(args, 2):
        if isinstance(args[2], FunctionHandle):
            raise MatlabError("robustfit: custom weight functions are not supported; use one of the named ones.")
        wfun = text(args[2]).lower()
    tune = scalar(args[3], "tune") if given(args, 3) else None
    const = True
    if given(args, 4):
        const = as_bool(args[4], "const")
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    D = np.column_stack([np.ones(X.shape[0]), X]) if const else X
    n, p = int(ok.sum()), D.shape[1]
    if n <= p:
        raise MatlabError("robustfit: there are not enough observations for the number of coefficients.")
    b, w, info = robust_irls(D[ok], y[ok], wfun, tune)
    if nargout <= 1:
        return b.reshape(-1, 1)
    rs, os_ = info["robust_s"], info["ols_s"]
    s = max(rs, np.sqrt((os_ ** 2 * p ** 2 + rs ** 2 * n) / (p ** 2 + n))) if wfun != "ols" else os_
    cov = s ** 2 * info["xtxi"]
    se = np.sqrt(np.diag(cov))
    with np.errstate(all="ignore"):
        t = b / se
        corr = cov / np.outer(se, se)

    def full(v):
        out = np.full(y.size, np.nan)
        out[ok] = v
        return out.reshape(-1, 1)

    stats = struct(ols_s=os_, robust_s=rs, mad_s=info["mad_s"], s=float(s), resid=full(info["resid"]),
                   rstud=full(info["resid"] * info["adj"] / s), se=se.reshape(-1, 1), covb=cov, coeffcorr=corr,
                   t=t.reshape(-1, 1), p=(2 * st.t.sf(np.abs(t), n - p)).reshape(-1, 1), w=full(w),
                   h=full(info["h"]), dfe=float(n - p))
    return b.reshape(-1, 1), stats


# -- nonlinear regression ---------------------------------------------------------------------------------------

def _statset_num(opts_struct: Any, name: str, default: float) -> float:
    if isinstance(opts_struct, Struct) and name in opts_struct.fields:
        v = opts_struct.get_field(0, name)
        if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "iuf":
            return float(v.flat[0])
    return default


def nl_fit(fmodel: Callable, y: np.ndarray, beta0: np.ndarray, w: np.ndarray | None, options: Any = None):
    """Nonlinear least squares by Levenberg-Marquardt; returns beta, weighted residuals and Jacobian."""
    from scipy.optimize import least_squares
    sw = np.ones(y.size) if w is None else np.sqrt(w)
    ok = ~np.isnan(y) & ~np.isnan(sw)

    def res(b):
        f = np.asarray(fmodel(b), dtype=float).flatten(order="F")
        if f.size != y.size:
            raise MatlabError("nlinfit: the model function must return one value per observation.")
        r = (y - f) * sw
        r = np.where(ok, r, 0.0)
        return np.where(np.isfinite(r), r, 1e150)

    if int(ok.sum()) < beta0.size:
        raise MatlabError("nlinfit: there are fewer observations than coefficients.")
    maxit = int(_statset_num(options, "MaxIter", 400))
    tolx = min(_statset_num(options, "TolX", 1e-8), 1e-8)
    tolf = min(_statset_num(options, "TolFun", 1e-8), 1e-8)
    try:
        sol = least_squares(res, beta0, method="lm", xtol=tolx * 1e-4, ftol=tolf * 1e-4, gtol=1e-14,
                            max_nfev=maxit * (beta0.size + 1) * 4, x_scale="jac")
        beta = sol.x
    except ValueError as e:
        raise MatlabError(f"nlinfit: {e}") from None
    R = res(beta)
    J = -num_jacobian(res, beta)
    return beta, R, J, ok


def _robust_requested(options: Any) -> bool:
    if not isinstance(options, Struct):
        return False
    for name in ("Robust", "RobustWgtFun"):
        if name in options.fields:
            v = options.get_field(0, name)
            if is_text(v) and text(v).lower() not in ("off", ""):
                return True
            if isinstance(v, FunctionHandle):
                return True
    return False


def _nl_model(interp, fun: Any, X: Any, shape: tuple) -> Callable:
    def f(b):
        return np.asarray(fnum(call(interp, fun, [np.asarray(b, dtype=float).reshape(shape), X], 1)[0]), dtype=float)
    return f


@builtin("nlinfit")
def _nlinfit(interp, args, nargout):
    if len(args) < 4:
        raise MatlabError("nlinfit: expected nlinfit(X, Y, modelfun, beta0).")
    pos, opts = get_opts(args, 4, ["Weights", "ErrorModel", "ErrorParameters", "Options"], "nlinfit")
    if len(pos) > 5:
        raise MatlabError("Too many input arguments for nlinfit.")
    options = opts.get("options", pos[4] if len(pos) > 4 else None)
    if options is not None and not is_empty_arg(options) and not isinstance(options, Struct):
        raise MatlabError("nlinfit: the options argument must be a structure created by statset.")
    if _robust_requested(options):
        raise MatlabError("nlinfit: robust fitting options are not supported.")
    if "errormodel" in opts and choice(opts["errormodel"], ["constant", "proportional", "combined"],
                                       "ErrorModel") != "constant":
        raise MatlabError("nlinfit: only the 'constant' error model is supported.")
    X, fun = pos[0], pos[2]
    y = vec(pos[1])
    b0 = np.asarray(fnum(pos[3]), dtype=float)
    w = None
    if "weights" in opts and not is_empty_arg(opts["weights"]):
        if isinstance(opts["weights"], FunctionHandle):
            raise MatlabError("nlinfit: weights given as a function handle are not supported.")
        w = vec(opts["weights"])
        if w.size != y.size or np.any(w < 0):
            raise MatlabError("nlinfit: Weights must hold one non-negative value per observation.")
    beta, R, J, ok = nl_fit(_nl_model(interp, fun, X, b0.shape), y, b0.flatten(order="F"), w, options)
    n, p = int(ok.sum()), beta.size
    mse = float(np.sum(R ** 2) / (n - p)) if n > p else np.nan
    with np.errstate(all="ignore"):
        cov = np.linalg.pinv(J.T @ J) * mse
    info = struct(ErrorModel="constant", ErrorParameters=float(np.sqrt(mse)), ErrorVariance=float(mse), MSE=mse,
                  ScheffeSimPred=float(p), WeightFunction=False, FixedWeights=w is not None,
                  RobustWeightFunction=False)
    R_out = np.where(ok, R, np.nan)
    return outs(nargout, beta.reshape(b0.shape, order="F"), R_out.reshape(-1, 1), J, cov, mse, info)


@builtin("nlparci")
def _nlparci(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 4:
        raise MatlabError("nlparci: expected nlparci(beta, resid, 'covar', sigma) or "
                          "nlparci(beta, resid, 'jacobian', J).")
    pos, opts = get_opts(args, 2, ["covar", "jacobian", "alpha"], "nlparci")
    if len(pos) != 2:
        raise MatlabError("nlparci: expected nlparci(beta, resid, 'covar', sigma) or (..., 'jacobian', J).")
    beta = vec(pos[0])
    r = vec(pos[1])
    r = r[~np.isnan(r)]
    alpha = scalar(opts["alpha"], "alpha") if "alpha" in opts else 0.05
    p = beta.size
    dfe = r.size - p
    if dfe <= 0:
        raise MatlabError("nlparci: there are not enough residuals for the number of coefficients.")
    if "covar" in opts:
        se = np.sqrt(np.clip(np.diag(np.asarray(fnum(opts["covar"]), dtype=float)), 0, None))
    elif "jacobian" in opts:
        J = np.asarray(fnum(opts["jacobian"]), dtype=float)
        J = J[~np.isnan(J).any(axis=1)]
        se = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J)), 0, None) * np.sum(r ** 2) / dfe)
    else:
        raise MatlabError("nlparci: give the coefficient covariance ('covar') or the Jacobian ('jacobian').")
    c = st.t.ppf(1 - alpha / 2, dfe)
    return np.column_stack([beta - c * se, beta + c * se])


@builtin("nlpredci")
def _nlpredci(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 6:
        raise MatlabError("nlpredci: expected nlpredci(modelfun, X, beta, resid, 'Covar', sigma) or "
                          "(..., 'Jacobian', J).")
    pos, opts = get_opts(args, 4, ["Covar", "Jacobian", "Alpha", "MSE", "PredOpt", "SimOpt", "ErrorModelInfo",
                                   "Weights"], "nlpredci")
    if len(pos) != 4:
        raise MatlabError("nlpredci: expected nlpredci(modelfun, X, beta, resid, 'Name', value, ...).")
    for o in ("errormodelinfo", "weights"):
        if o in opts:
            raise MatlabError(f"nlpredci: the option '{o}' is not supported.")
    fun, X = pos[0], pos[1]
    b = np.asarray(fnum(pos[2]), dtype=float)
    r = vec(pos[3])
    r = r[~np.isnan(r)]
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    obs = "predopt" in opts and choice(opts["predopt"], ["curve", "observation"], "PredOpt") == "observation"
    simul = "simopt" in opts and as_bool(opts["simopt"], "SimOpt")
    p = b.size
    dfe = r.size - p
    mse = scalar(opts["mse"], "MSE") if "mse" in opts else float(np.sum(r ** 2) / dfe)
    if "covar" in opts:
        cov = np.asarray(fnum(opts["covar"]), dtype=float)
    elif "jacobian" in opts:
        J = np.asarray(fnum(opts["jacobian"]), dtype=float)
        J = J[~np.isnan(J).any(axis=1)]
        cov = np.linalg.pinv(J.T @ J) * mse
    else:
        raise MatlabError("nlpredci: give the coefficient covariance ('Covar') or the Jacobian ('Jacobian').")
    f = _nl_model(interp, fun, X, b.shape)
    ypred = f(b.flatten(order="F"))
    shape = ypred.shape
    Jx = num_jacobian(lambda t: f(t).flatten(order="F"), b.flatten(order="F"))
    var = np.einsum("ij,jk,ik->i", Jx, cov, Jx) + (mse if obs else 0.0)
    if simul:
        k = p + (1 if obs else 0)
        crit = np.sqrt(k * st.f.ppf(1 - alpha, k, dfe))
    else:
        crit = st.t.ppf(1 - alpha / 2, dfe)
    delta = crit * np.sqrt(np.clip(var, 0, None))
    return outs(nargout, ypred, delta.reshape(shape, order="F"))


class NonLinearModel(StatObject):
    matlab_class = "NonLinearModel"

    def __init__(self, interp, fun, X, y, beta0, cnames, pnames, rname, w, options, formula: str, tbl_vars=None):
        super().__init__()
        self.fun, self.X, self.y, self.w = fun, X, y, w
        self.cnames, self.pnames, self.rname, self.formula = cnames, pnames, rname, formula
        self.shape = beta0.shape
        self.tbl_vars = tbl_vars
        beta, R, J, ok = nl_fit(_nl_model(interp, fun, X, self.shape), y, beta0.flatten(order="F"), w, options)
        self.beta, self.J, self.ok = beta, J, ok
        sw = np.ones(y.size) if w is None else np.sqrt(w)
        self.fitted = _nl_model(interp, fun, X, self.shape)(beta).flatten(order="F")
        self.resid = np.where(ok, y - self.fitted, np.nan)
        self.n, self.p = int(ok.sum()), beta.size
        self.dfe = self.n - self.p
        self.sse = float(np.nansum((self.resid * sw) ** 2))
        self.mse = self.sse / self.dfe if self.dfe > 0 else np.nan
        with np.errstate(all="ignore"):
            self.cov = np.linalg.pinv(J.T @ J) * self.mse
            self.se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
            self.t = beta / self.se
        wv = np.ones(y.size) if w is None else w
        yb = np.nansum(wv[ok] * y[ok]) / np.sum(wv[ok])
        self.sst = float(np.sum(wv[ok] * (y[ok] - yb) ** 2))

    def property_names(self) -> list[str]:
        return ["Coefficients", "CoefficientNames", "CoefficientCovariance", "NumCoefficients", "NumObservations",
                "DFE", "MSE", "RMSE", "SSE", "SSR", "SST", "Rsquared", "LogLikelihood", "ModelCriterion", "Fitted",
                "Residuals", "Formula", "PredictorNames", "ResponseName", "NumPredictors"]

    def _p(self):
        import scipy.stats as st
        with np.errstate(all="ignore"):
            return 2 * st.t.sf(np.abs(self.t), self.dfe) if self.dfe > 0 else np.full(self.t.shape, np.nan)

    def get_field(self, interp, name: str) -> Any:
        if name == "Coefficients":
            return _coef_table(self.cnames, self.beta, self.se, self.t, self._p())
        if name == "CoefficientNames":
            return cellstr(self.cnames, column=False)
        if name == "CoefficientCovariance":
            return self.cov.copy()
        if name == "NumCoefficients":
            return float(self.p)
        if name == "NumObservations":
            return float(self.n)
        if name == "NumPredictors":
            return float(len(self.pnames))
        if name in ("DFE", "MSE", "SSE", "SST"):
            return float(getattr(self, name.lower()))
        if name == "SSR":
            return float(self.sst - self.sse)
        if name == "RMSE":
            return float(np.sqrt(self.mse))
        if name == "Rsquared":
            with np.errstate(all="ignore"):
                r2 = 1 - self.sse / self.sst
                adj = 1 - (self.sse / self.dfe) / (self.sst / (self.n - 1)) if self.dfe > 0 else np.nan
            return struct(Ordinary=float(r2), Adjusted=float(adj))
        if name == "LogLikelihood":
            return float(-self.n / 2 * (np.log(2 * np.pi) + np.log(self.sse / self.n) + 1)) if self.sse > 0 else np.inf
        if name == "ModelCriterion":
            ll = self.get_field(interp, "LogLikelihood")
            ll = float(np.asarray(ll).flat[0])
            k, n = self.p, self.n
            aic = -2 * ll + 2 * k
            return struct(AIC=aic, AICc=aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan,
                          BIC=-2 * ll + k * np.log(n), CAIC=-2 * ll + k * (np.log(n) + 1))
        if name == "Fitted":
            return np.where(self.ok, self.fitted, np.nan).reshape(-1, 1)
        if name == "Residuals":
            with np.errstate(all="ignore"):
                rmse = np.sqrt(self.mse)
                h = np.einsum("ij,jk,ik->i", self.J, np.linalg.pinv(self.J.T @ self.J), self.J)
                std = self.resid / (rmse * np.sqrt(1 - h))
            return make_table(["Raw", "Pearson", "Standardized"],
                              [self.resid.reshape(-1, 1), (self.resid / rmse).reshape(-1, 1), std.reshape(-1, 1)])
        if name == "Formula":
            return char(self.formula)
        if name == "PredictorNames":
            return cellstr(self.pnames)
        if name == "ResponseName":
            return char(self.rname)
        return super().get_field(interp, name)

    def _xnew(self, Xnew: Any) -> Any:
        if is_table(Xnew):
            cols = dict(zip(Xnew.names, Xnew.columns))
            missing = [n for n in self.pnames if n not in cols]
            if missing:
                raise MatlabError(f"predict: the table has no variable '{missing[0]}'.")
            return np.column_stack([vec(cols[n]) for n in self.pnames])
        return Xnew

    def m_predict(self, interp, args, nargout):
        import scipy.stats as st
        pos, opts = get_opts(args, 0, ["Alpha", "Prediction", "Simultaneous", "Weights"], "predict")
        if len(pos) > 1:
            raise MatlabError("Too many input arguments for predict.")
        if "weights" in opts:
            raise MatlabError("predict: the option 'Weights' is not supported.")
        X = self._xnew(pos[0]) if pos else self.X
        f = _nl_model(interp, self.fun, X, self.shape)
        ypred = f(self.beta).flatten(order="F")
        if nargout < 2:
            return ypred.reshape(-1, 1)
        alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
        obs = "prediction" in opts and choice(opts["prediction"], ["curve", "observation"],
                "Prediction") == "observation"
        simul = "simultaneous" in opts and as_bool(opts["simultaneous"], "Simultaneous")
        Jx = num_jacobian(lambda t: f(t).flatten(order="F"), self.beta)
        var = np.einsum("ij,jk,ik->i", Jx, self.cov, Jx) + (self.mse if obs else 0.0)
        if simul:
            k = self.p + (1 if obs else 0)
            crit = np.sqrt(k * st.f.ppf(1 - alpha, k, self.dfe))
        else:
            crit = st.t.ppf(1 - alpha / 2, self.dfe)
        half = crit * np.sqrt(np.clip(var, 0, None))
        return ypred.reshape(-1, 1), np.column_stack([ypred - half, ypred + half])

    def m_feval(self, interp, args, nargout):
        if not args:
            raise MatlabError("feval: the predictor values are missing.")
        X = self._xnew(args[0]) if len(args) == 1 else np.column_stack([vec(a) for a in args])
        return _nl_model(interp, self.fun, X, self.shape)(self.beta).reshape(-1, 1)

    def m_random(self, interp, args, nargout):
        X = self._xnew(args[0]) if args else self.X
        mu = _nl_model(interp, self.fun, X, self.shape)(self.beta).flatten(order="F")
        return (mu + np.sqrt(self.mse) * rng_of(interp).standard_normal(mu.size)).reshape(-1, 1)

    def m_coefCI(self, interp, args, nargout):
        import scipy.stats as st
        alpha = scalar(args[0], "alpha") if args else 0.05
        c = st.t.ppf(1 - alpha / 2, self.dfe)
        return np.column_stack([self.beta - c * self.se, self.beta + c * self.se])

    def m_coefTest(self, interp, args, nargout):
        import scipy.stats as st
        p = self.p
        H = np.atleast_2d(np.asarray(fnum(args[0]), dtype=float)) if args else np.eye(p)
        c = vec(args[1]) if len(args) > 1 else np.zeros(H.shape[0])
        d = H @ self.beta - c
        V = H @ self.cov @ H.T
        r = int(np.linalg.matrix_rank(V))
        F = float(d @ np.linalg.pinv(V) @ d / r) if r else np.nan
        return outs(nargout, float(st.f.sf(F, r, self.dfe)) if r else np.nan, F, float(r))

    def display_lines(self) -> list[str]:
        import scipy.stats as st
        lines = ["", "Nonlinear regression model:", "    " + self.formula, "", "Estimated Coefficients:"]
        lines += _coef_lines(self.cnames, {"Estimate": self.beta, "SE": self.se, "tStat": self.t, "pValue": self._p()})
        with np.errstate(all="ignore"):
            r2 = 1 - self.sse / self.sst
            adj = 1 - (self.sse / self.dfe) / (self.sst / (self.n - 1)) if self.dfe > 0 else np.nan
            F = ((np.nansum(self.y[self.ok] ** 2) - self.sse) / self.p) / self.mse
        lines += ["", "", f"Number of observations: {self.n}, Error degrees of freedom: {self.dfe}",
                  f"Root Mean Squared Error: {_fmt_g(float(np.sqrt(self.mse)), 3)}",
                  f"R-Squared: {_fmt_g(float(r2), 3)},  Adjusted R-Squared {_fmt_g(float(adj), 3)}",
                  f"F-statistic vs. zero model: {_fmt_g(float(F), 3)}, "
                  f"p-value = {_fmt_g(float(st.f.sf(F, self.p, self.dfe)), 3)}"]
        return lines

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


def _formula_handle(interp, formula: str, pnames: list[str], all_names: list[str]):
    """Turn 'y ~ b1*exp(b2*x)' into a handle @(b, X) and the coefficient names."""
    lhs, rhs = formula.split("~", 1)
    resp = lhs.strip()
    idents = re.findall(r"(?<![\w.])([A-Za-z_]\w*)\b(?!\s*\()", rhs)
    preds = [n for n in dict.fromkeys(idents) if n in pnames]
    coefs = [n for n in dict.fromkeys(idents) if n not in all_names and n not in ("pi", "exp", "Inf", "NaN", "eps")]
    expr = rhs

    def repl(m):
        nme = m.group(1)
        if nme in coefs:
            return f"b__({coefs.index(nme) + 1})"
        if nme in preds:
            return f"X__(:,{preds.index(nme) + 1})"
        return nme
    expr = re.sub(r"(?<![\w.])([A-Za-z_]\w*)\b(?!\s*\()", repl, expr)
    expr = re.sub(r"(?<!\.)([*/^])", r".\1", expr)
    from .ndmath import _eval_handle
    return _eval_handle(interp, f"@(b__, X__) {expr}"), coefs, preds, resp


@builtin("fitnlm")
def _fitnlm(interp, args, nargout):
    known = ["CoefficientNames", "ErrorModel", "ErrorParameters", "Exclude", "Options", "PredictorVars",
             "ResponseVar", "VarNames", "Weights"]
    pos, opts = get_opts(args, 2, known, "fitnlm", text_pos=(2,))
    if "errormodel" in opts and choice(opts["errormodel"], ["constant", "proportional", "combined"],
                                       "ErrorModel") != "constant":
        raise MatlabError("fitnlm: only the 'constant' error model is supported.")
    options = opts.get("options")
    if _robust_requested(options):
        raise MatlabError("fitnlm: robust fitting options are not supported.")
    if is_table(pos[0]):
        if len(pos) != 3:
            raise MatlabError("fitnlm: expected fitnlm(tbl, modelfun, beta0).")
        tbl, fun, b0 = pos
        names = list(tbl.names)
        cols = dict(zip(tbl.names, tbl.columns))
        resp = text(opts["responsevar"]) if "responsevar" in opts else names[-1]
        pnames = text_list(opts["predictorvars"]) if "predictorvars" in opts else [n for n in names if n != resp]
        if is_text(fun):
            fun, coefs, pnames, resp = _formula_handle(interp, text(fun), [n for n in names], names)
            formula = text(pos[1]).strip()
        else:
            coefs, formula = None, None
        if resp not in cols:
            raise MatlabError(f"fitnlm: '{resp}' is not a variable of the table.")
        X = np.column_stack([vec(cols[n]) for n in pnames]) if pnames else np.zeros((tbl.nrows, 0))
        y = vec(cols[resp])
    else:
        if len(pos) != 4:
            raise MatlabError("fitnlm: expected fitnlm(X, y, modelfun, beta0).")
        X, y, fun, b0 = pos[0], vec(pos[1]), pos[2], pos[3]
        Xa = np.asarray(fnum(X), dtype=float)
        p = Xa.shape[1] if Xa.ndim == 2 else 1
        vn = text_list(opts["varnames"]) if "varnames" in opts else [f"x{j + 1}" for j in range(p)] + ["y"]
        if len(vn) != p + 1:
            raise MatlabError(f"fitnlm: VarNames must list {p + 1} names.")
        pnames, resp = vn[:-1], vn[-1]
        if is_text(fun):
            fun, coefs, used, resp = _formula_handle(interp, text(fun), pnames, vn)
            X = Xa[:, [pnames.index(u) for u in used]]
            pnames = used
            formula = text(pos[2]).strip()
        else:
            coefs, formula = None, None
    b0 = np.asarray(fnum(b0), dtype=float)
    if coefs is not None and len(coefs) != b0.size:
        raise MatlabError(f"fitnlm: the formula has {len(coefs)} coefficients but beta0 has {b0.size} values.")
    cnames = coefs or [f"b{i + 1}" for i in range(b0.size)]
    if "coefficientnames" in opts:
        cnames = text_list(opts["coefficientnames"])
        if cnames is None or len(cnames) != b0.size:
            raise MatlabError("fitnlm: CoefficientNames must list one name per coefficient.")
    w = None
    if "weights" in opts:
        w = vec(opts["weights"])
        if w.size != y.size:
            raise MatlabError("fitnlm: Weights must hold one value per observation.")
    if "exclude" in opts:
        y = y.copy()
        y[_row_mask(opts["exclude"], y.size, "Exclude")] = np.nan
    if not isinstance(fun, FunctionHandle):
        raise MatlabError("fitnlm: the model must be a function handle or a formula.")
    if formula is None:
        formula = f"{resp} ~ " + (repr(fun) if fun.is_anonymous else f"{fun.name}(b, X)")
    return NonLinearModel(interp, fun, X, y, b0, cnames, pnames, resp, w, options, formula)


# -- lasso, elastic net, ridge -----------------------------------------------------------------------------------------

def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _cd_path(X: np.ndarray, y: np.ndarray, w: np.ndarray, lambdas: np.ndarray, alpha: float, tol: float,
             max_iter: int, dfmax: float = np.inf) -> np.ndarray:
    """Elastic-net coordinate descent for (1/2) sum w (y - X b)^2 + lam (alpha |b|_1 + (1 - alpha)/2 |b|^2).

    X and y are already centred (weighted); w sums to one. Returns one column of coefficients per lambda.
    """
    n, p = X.shape
    B = np.zeros((p, lambdas.size))
    b = np.zeros(p)
    r = y.copy()
    xsq = (w[:, None] * X ** 2).sum(axis=0)
    Xw = X * w[:, None]
    for k, lam in enumerate(lambdas):
        for it in range(max_iter):
            delta = 0.0
            active = range(p) if it % 10 == 0 or it < 2 else np.flatnonzero(b != 0)
            for j in active:
                if xsq[j] == 0:
                    continue
                rho = Xw[:, j] @ r + xsq[j] * b[j]
                new = _soft(rho, lam * alpha) / (xsq[j] + lam * (1 - alpha))
                if new != b[j]:
                    r -= X[:, j] * (new - b[j])
                    delta = max(delta, abs(new - b[j]) * np.sqrt(xsq[j]))
                    b[j] = new
            if delta <= tol * max(np.sqrt(float(w @ y ** 2)), 1e-300) and (it % 10 == 0 or it < 2):
                break
        B[:, k] = b
        if np.count_nonzero(b) > dfmax:
            B[:, k + 1:] = np.nan
            break
    return B


def _lasso_prepare(X, y, w, standardize: bool, intercept: bool):
    mx = w @ X if intercept else np.zeros(X.shape[1])
    my = float(w @ y) if intercept else 0.0
    X0 = X - mx
    sx = np.sqrt((w[:, None] * X0 ** 2).sum(axis=0)) if standardize else np.ones(X.shape[1])
    sx = np.where(sx > 0, sx, 1.0)
    return X0 / sx, y - my, mx, my, sx


def _lambda_path(Xs, yc, w, alpha, nlam, ratio) -> np.ndarray:
    lam_max = float(np.max(np.abs((Xs * w[:, None]).T @ yc))) / max(alpha, 1e-3)
    if lam_max <= 0:
        return np.zeros(1)
    if nlam == 1:
        return np.array([lam_max])
    return lam_max * ratio ** (np.arange(nlam) / (nlam - 1.0))


def lasso_fit(X, y, w, lambdas, alpha, standardize, intercept, tol, max_iter, nlam, ratio, dfmax=np.inf):
    """Coefficients on the original scale (columns follow ascending lambda) and the fit information."""
    Xs, yc, mx, my, sx = _lasso_prepare(X, y, w, standardize, intercept)
    if lambdas is None:
        lam = _lambda_path(Xs, yc, w, alpha, nlam, ratio)
    else:
        lam = np.sort(np.asarray(lambdas, dtype=float))[::-1]
    Bs = _cd_path(Xs, yc, w, lam, alpha, tol, max_iter, dfmax)
    keep = ~np.isnan(Bs).any(axis=0)
    Bs, lam = Bs[:, keep], lam[keep]
    B = Bs / sx[:, None]
    b0 = my - mx @ B
    order = np.argsort(lam, kind="mergesort")
    return B[:, order], b0[order], lam[order]


_LASSO_OPTS = ["AbsTol", "Alpha", "B0", "CV", "DFmax", "Intercept", "Lambda", "LambdaRatio", "MaxIter", "MCReps",
               "NumLambda", "Options", "PredictorNames", "RelTol", "Standardize", "Weights", "CacheSize", "Rho",
               "U0", "UseCovariance"]


def _cv_folds(interp, opts: dict, n: int, fname: str):
    """Train/test index pairs for the 'CV' option (None for resubstitution)."""
    from .stats_desc import CVPartition, make_partition
    if "cv" not in opts:
        return None
    v = opts["cv"]
    if is_text(v):
        if text(v).lower() != "resubstitution":
            raise MatlabError(f"{fname}: CV must be 'resubstitution', a number of folds or a cvpartition.")
        return None
    if isinstance(v, CVPartition):
        if v.n != n:
            raise MatlabError(f"{fname}: the partition does not match the number of observations.")
        return v.train_test()
    return make_partition(rng_of(interp), n, "kfold", intval(v, "CV"), None).train_test()


@builtin("lasso")
def _lasso(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("lasso: expected lasso(X, y, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, _LASSO_OPTS, "lasso")
    if len(pos) != 2:
        raise MatlabError("lasso: expected lasso(X, y, 'Name', value, ...).")
    for o in ("b0", "u0", "rho", "abstol"):
        if o in opts:
            raise MatlabError(f"lasso: the option '{o}' is not supported.")
    X = np.asarray(fnum(pos[0]), dtype=float)
    y = vec(pos[1])
    if X.ndim != 2 or X.shape[0] != y.size:
        raise MatlabError("lasso: X must have one row per element of y.")
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    n_all, p = X.shape
    w = np.ones(n_all)
    if "weights" in opts:
        w = vec(opts["weights"])
        if w.size != n_all or np.any(w < 0):
            raise MatlabError("lasso: Weights must hold one non-negative value per observation.")
        ok &= ~np.isnan(w)
    X, y, w = X[ok], y[ok], w[ok]
    n = y.size
    w = w / w.sum()
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 1.0
    if not 0 < alpha <= 1:
        raise MatlabError("lasso: Alpha must be in the interval (0, 1].")
    standardize = as_bool(opts["standardize"], "Standardize") if "standardize" in opts else True
    intercept = as_bool(opts["intercept"], "Intercept") if "intercept" in opts else True
    nlam = intval(opts["numlambda"], "NumLambda") if "numlambda" in opts else 100
    ratio = scalar(opts["lambdaratio"], "LambdaRatio") if "lambdaratio" in opts else (1e-4 if n > p else 1e-2)
    reltol = scalar(opts["reltol"], "RelTol") if "reltol" in opts else 1e-4
    max_iter = intval(opts["maxiter"], "MaxIter") if "maxiter" in opts else 100000
    dfmax = scalar(opts["dfmax"], "DFmax") if "dfmax" in opts else np.inf
    lambdas = vec(opts["lambda"]) if "lambda" in opts and not is_empty_arg(opts["lambda"]) else None
    if lambdas is not None and np.any(lambdas < 0):
        raise MatlabError("lasso: Lambda values must not be negative.")
    tol = min(reltol, 1e-4) * 1e-3
    B, b0, lam = lasso_fit(X, y, w, lambdas, alpha, standardize, intercept, tol, max_iter, nlam, ratio, dfmax)
    names = text_list(opts["predictornames"]) if "predictornames" in opts else None
    if names is not None and len(names) != p:
        raise MatlabError("lasso: PredictorNames must list one name per column of X.")
    mse = np.array([float(w @ (y - b0[k] - X @ B[:, k]) ** 2) for k in range(lam.size)])
    info = {"Intercept": b0.reshape(1, -1), "Lambda": lam.reshape(1, -1), "Alpha": alpha,
            "DF": np.count_nonzero(B, axis=0).astype(float).reshape(1, -1), "MSE": mse.reshape(1, -1),
            "PredictorNames": cellstr(names, column=False) if names else cellstr([], column=False)}
    folds = _cv_folds(interp, opts, n, "lasso")
    if folds is not None:
        reps = intval(opts["mcreps"], "MCReps") if "mcreps" in opts else 1
        if reps != 1:
            raise MatlabError("lasso: MCReps other than 1 is not supported.")
        fold_mse = np.empty((len(folds), lam.size))
        for f, (tr, te) in enumerate(folds):
            wt = w[tr] / w[tr].sum()
            Bf, b0f, _ = lasso_fit(X[tr], y[tr], wt, lam, alpha, standardize, intercept, tol, max_iter, nlam, ratio)
            err = y[te, None] - b0f[None, :] - X[te] @ Bf
            fold_mse[f] = (w[te, None] * err ** 2).sum(axis=0) / w[te].sum()
        cv_mse = fold_mse.mean(axis=0)
        se = fold_mse.std(axis=0, ddof=1) / np.sqrt(len(folds)) if len(folds) > 1 else np.zeros(lam.size)
        imin = int(np.argmin(cv_mse))
        within = np.flatnonzero(cv_mse <= cv_mse[imin] + se[imin])
        i1se = int(within.max())
        info.update({"MSE": cv_mse.reshape(1, -1), "SE": se.reshape(1, -1), "LambdaMinMSE": float(lam[imin]),
                     "Lambda1SE": float(lam[i1se]), "IndexMinMSE": float(imin + 1), "Index1SE": float(i1se + 1)})
    return outs(nargout, B, struct(**info))


@builtin("ridge")
def _ridge(interp, args, nargout):
    if not 3 <= len(args) <= 4:
        raise MatlabError("ridge: expected ridge(y, X, k) or ridge(y, X, k, scaled).")
    y = vec(args[0])
    X = np.asarray(fnum(args[1]), dtype=float)
    ks = vec(args[2])
    scaled = bool(scalar(args[3], "scaled")) if given(args, 3) else True
    if X.ndim != 2 or X.shape[0] != y.size:
        raise MatlabError("ridge: X must have one row per element of y.")
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[ok], y[ok]
    n, p = X.shape
    mx = X.mean(axis=0)
    sx = X.std(axis=0, ddof=1)
    sx = np.where(sx > 0, sx, 1.0)
    Z = (X - mx) / sx
    out = []
    for k in ks:
        A = np.vstack([Z, np.sqrt(k) * np.eye(p)])
        t = np.concatenate([y - y.mean(), np.zeros(p)])
        b, *_ = np.linalg.lstsq(A, t, rcond=None)
        if scaled:
            out.append(b)
        else:
            bo = b / sx
            out.append(np.concatenate([[y.mean() - mx @ bo], bo]))
    return np.column_stack(out)


@builtin("lassoglm")
def _lassoglm(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("lassoglm: expected lassoglm(X, y, distr, 'Name', value, ...).")
    known = ["Alpha", "CV", "DFmax", "Lambda", "LambdaRatio", "Link", "MaxIter", "MCReps", "NumLambda", "Offset",
             "Options", "PredictorNames", "RelTol", "Standardize", "Weights"]
    pos, opts = get_opts(args, 2, known, "lassoglm", text_pos=(2,))
    if len(pos) > 3:
        raise MatlabError("Too many input arguments for lassoglm.")
    for o in ("cv", "mcreps"):
        if o in opts:
            raise MatlabError(f"lassoglm: the option '{o}' is not supported.")
    dist = _dist_name(pos[2]) if len(pos) > 2 and not is_empty_arg(pos[2]) else "normal"
    X, y, N = _glm_xy(pos[0], pos[1], "lassoglm")
    n, p = X.shape
    link = Link(opts.get("link"), dist)
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 1.0
    if not 0 < alpha <= 1:
        raise MatlabError("lassoglm: Alpha must be in the interval (0, 1].")
    standardize = as_bool(opts["standardize"], "Standardize") if "standardize" in opts else True
    nlam = intval(opts["numlambda"], "NumLambda") if "numlambda" in opts else 100
    ratio = scalar(opts["lambdaratio"], "LambdaRatio") if "lambdaratio" in opts else (1e-4 if n > p else 1e-2)
    dfmax = scalar(opts["dfmax"], "DFmax") if "dfmax" in opts else np.inf
    pw = vec(opts["weights"]) if "weights" in opts else np.ones(n)
    off = np.broadcast_to(vec(opts["offset"]), (n,)).astype(float) if "offset" in opts else np.zeros(n)
    if N is not None:
        with np.errstate(all="ignore"):
            y = np.where(N > 0, y / np.where(N > 0, N, 1), 0.0)
    else:
        N = np.ones(n)
    wn = pw / pw.sum()
    mx = wn @ X
    sx = np.sqrt(wn @ (X - mx) ** 2) if standardize else np.ones(p)
    sx = np.where(sx > 0, sx, 1.0)
    Xs = (X - mx) / sx
    null = GLMFit(np.ones((n, 1)), y, dist, link, N, pw, off, None)
    from .stats_models import _deviance_terms, _variance
    with np.errstate(all="ignore"):
        g0 = (y - null.mu) * (N if dist == "binomial" else 1.0) / (_variance(dist, null.mu) * link.deriv(null.mu))
    lam_max = float(np.max(np.abs((Xs * wn[:, None]).T @ g0))) / alpha if p else 0.0
    if "lambda" in opts and not is_empty_arg(opts["lambda"]):
        lam = np.sort(vec(opts["lambda"]))[::-1]
    else:
        lam = lam_max * ratio ** (np.arange(nlam) / max(nlam - 1.0, 1.0)) if lam_max > 0 else np.zeros(1)
    b0 = float(null.beta[0])
    b = np.zeros(p)
    B = np.full((p, lam.size), np.nan)
    B0 = np.full(lam.size, np.nan)
    dev = np.full(lam.size, np.nan)
    for k, lm in enumerate(lam):
        for _outer in range(100):
            eta = b0 + Xs @ b + off
            mu = link.inv(eta)
            mu = np.clip(mu, 1e-12, 1 - 1e-12) if dist == "binomial" else (np.maximum(mu, 1e-300) if dist != "normal"
                    else mu)
            with np.errstate(all="ignore"):
                d = link.deriv(mu)
                W = wn * (N if dist == "binomial" else 1.0) / (d ** 2 * _variance(dist, mu))
                z = eta - off + (y - mu) * d
            W = np.where(np.isfinite(W), W, 0.0)
            old_b, old_b0 = b.copy(), b0
            sw = W.sum()
            for _inner in range(1000):
                b0 = float(W @ (z - Xs @ b) / sw)
                r = z - b0 - Xs @ b
                delta = 0.0
                for j in range(p):
                    xj = Xs[:, j]
                    q = float(W @ xj ** 2)
                    if q == 0:
                        continue
                    rho = float((W * xj) @ r) + q * b[j]
                    new = _soft(rho, lm * alpha) / (q + lm * (1 - alpha))
                    if new != b[j]:
                        r -= xj * (new - b[j])
                        delta = max(delta, abs(new - b[j]))
                        b[j] = new
                if delta < 1e-9:
                    break
            if max(np.max(np.abs(b - old_b), initial=0.0), abs(b0 - old_b0)) < 1e-8:
                break
        eta = b0 + Xs @ b + off
        mu = link.inv(eta)
        if dist == "binomial":
            mu = np.clip(mu, 1e-15, 1 - 1e-15)
        B[:, k] = b / sx
        B0[k] = b0 - mx @ (b / sx)
        dev[k] = float(np.sum(pw * _deviance_terms(dist, y, mu, N)))
        if np.count_nonzero(b) > dfmax:
            break
    keep = ~np.isnan(B0)
    order = np.argsort(lam[keep], kind="mergesort")
    B, B0, dev, lam = B[:, keep][:, order], B0[keep][order], dev[keep][order], lam[keep][order]
    names = text_list(opts["predictornames"]) if "predictornames" in opts else []
    info = struct(Intercept=B0.reshape(1, -1), Lambda=lam.reshape(1, -1), Alpha=alpha,
                  DF=np.count_nonzero(B, axis=0).astype(float).reshape(1, -1), Deviance=dev.reshape(1, -1),
                  PredictorNames=cellstr(names or [], column=False))
    return outs(nargout, B, info)


# -- stepwisefit ---------------------------------------------------------------------------------------------

@builtin("stepwisefit")
def _stepwisefit(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 2:
        raise MatlabError("stepwisefit: expected stepwisefit(X, y, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["PEnter", "PRemove", "Display", "InModel", "Keep", "MaxIter", "Scale"],
                         "stepwisefit")
    if len(pos) != 2:
        raise MatlabError("stepwisefit: expected stepwisefit(X, y, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    y = vec(pos[1])
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[ok], y[ok]
    n, p = X.shape
    penter = scalar(opts["penter"], "PEnter") if "penter" in opts else 0.05
    premove = scalar(opts["premove"], "PRemove") if "premove" in opts else max(penter, 0.10)
    display = as_bool(opts["display"], "Display") if "display" in opts else True
    inmodel = np.zeros(p, dtype=bool)
    if "inmodel" in opts:
        inmodel = _row_mask(opts["inmodel"], p, "InModel")
    keep = _row_mask(opts["keep"], p, "Keep") if "keep" in opts else np.zeros(p, dtype=bool)
    maxit = intval(opts["maxiter"], "MaxIter") if "maxiter" in opts else 10 ** 6
    if "scale" in opts and as_bool(opts["scale"], "Scale"):
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    def fit(mask):
        D = np.column_stack([np.ones(n), X[:, mask]])
        return LSFit(D, y, None, True)

    def stats_for(mask):
        """Coefficient, standard error and p-value of every column given the current model."""
        cur = fit(mask)
        B, SE, P = np.zeros(p), np.zeros(p), np.ones(p)
        pv = cur.pvalues()
        B[mask], SE[mask], P[mask] = cur.beta[1:], cur.se[1:], pv[1:]
        for j in np.flatnonzero(~mask):
            m2 = mask.copy()
            m2[j] = True
            f2 = fit(m2)
            k = int(np.sum(m2[:j + 1]))
            B[j], SE[j], P[j] = f2.beta[k], f2.se[k], f2.pvalues()[k]
        return cur, B, SE, P

    history_in = []
    nextstep = 0
    for _ in range(maxit):
        cur, B, SE, P = stats_for(inmodel)
        cand = np.flatnonzero(~inmodel & ~keep)
        nextstep = 0
        if cand.size and np.nanmin(P[cand]) < penter:
            j = int(cand[np.nanargmin(P[cand])])
            inmodel[j] = True
            nextstep = j + 1
            if display:
                interp.out(f"Initial columns included: none\nStep {len(history_in) + 1}, added column {j + 1}, "
                           f"p={_fmt_g(float(P[j]))}\n" if not history_in else
                           f"Step {len(history_in) + 1}, added column {j + 1}, p={_fmt_g(float(P[j]))}\n")
            history_in.append(inmodel.copy())
            continue
        out = np.flatnonzero(inmodel & ~keep)
        if out.size and np.nanmax(P[out]) > premove:
            j = int(out[np.nanargmax(P[out])])
            inmodel[j] = False
            nextstep = j + 1
            if display:
                interp.out(f"Step {len(history_in) + 1}, removed column {j + 1}, p={_fmt_g(float(P[j]))}\n")
            history_in.append(inmodel.copy())
            continue
        break
    cur, B, SE, P = stats_for(inmodel)
    if display:
        cols = " ".join(str(j + 1) for j in np.flatnonzero(inmodel)) or "none"
        interp.out(f"Final columns included:  {cols} \n")
    with np.errstate(all="ignore"):
        df0 = int(inmodel.sum())
        F = ((cur.sst - cur.sse) / df0) / cur.mse if df0 else np.nan
        T = B / SE
    stats = struct(source="stepwisefit", dfe=float(cur.dfe), df0=float(df0), SStotal=cur.sst, SSresid=cur.sse,
                   fstat=float(F), pval=float(st.f.sf(F, df0, cur.dfe)) if df0 else np.nan,
                   rmse=float(np.sqrt(cur.mse)), B=B.reshape(-1, 1), SE=SE.reshape(-1, 1), TSTAT=T.reshape(-1, 1),
                   PVAL=P.reshape(-1, 1), intercept=float(cur.beta[0]), wasnan=(~ok).reshape(-1, 1))
    hist = struct(rmse=np.array([[float(np.sqrt(fit(m).mse)) for m in history_in]]),
                  df0=np.array([[float(m.sum()) for m in history_in]]),
                  **{"in": np.array(history_in, dtype=bool).reshape(len(history_in), p)})
    return outs(nargout, B.reshape(-1, 1), SE.reshape(-1, 1), P.reshape(-1, 1), inmodel.reshape(1, -1), stats,
                float(nextstep), hist)


# -- polyconf ------------------------------------------------------------------------------------------------

@builtin("polyconf")
def _polyconf(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 2:
        raise MatlabError("polyconf: expected polyconf(p, x) or polyconf(p, x, S, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["alpha", "mu", "predopt", "simopt"], "polyconf")
    if len(pos) > 3:
        if len(pos) == 4 and not opts:
            opts["alpha"] = pos[3]          # older form: polyconf(p, x, S, alpha)
            pos = pos[:3]
        else:
            raise MatlabError("Too many input arguments for polyconf.")
    p = vec(pos[0])
    x = np.asarray(fnum(pos[1]), dtype=float)
    mu = vec(opts["mu"]) if "mu" in opts else None
    xs = (x - mu[0]) / mu[1] if mu is not None else x
    y = np.polyval(p, xs)
    if nargout < 2:
        return y
    if len(pos) < 3 or not isinstance(pos[2], Struct):
        raise MatlabError("polyconf: the error estimate needs the structure S returned by polyfit.")
    S = pos[2]
    alpha = scalar(opts["alpha"], "alpha") if "alpha" in opts else 0.05
    obs = not ("predopt" in opts and choice(opts["predopt"], ["observation", "curve"], "predopt") == "curve")
    simul = "simopt" in opts and as_bool(opts["simopt"], "simopt")
    R = np.asarray(fnum(S.get_field(0, "R")), dtype=float)
    df = scalar(S.get_field(0, "df"), "S.df")
    normr = scalar(S.get_field(0, "normr"), "S.normr")
    k = p.size
    V = np.vander(xs.flatten(order="F"), k)
    E = np.linalg.solve(R.T, V.T).T
    with np.errstate(all="ignore"):
        e2 = np.sum(E ** 2, axis=1) + (1.0 if obs else 0.0)
        if simul:
            kk = k + (1 if obs else 0)
            crit = np.sqrt(kk * st.f.ppf(1 - alpha, kk, df))
        else:
            crit = st.t.ppf(1 - alpha / 2, df)
        delta = crit * normr / np.sqrt(df) * np.sqrt(e2)
    return y, delta.reshape(x.shape, order="F")


# -- partial least squares -----------------------------------------------------------------------------------

def _simpls(X0: np.ndarray, Y0: np.ndarray, ncomp: int):
    n, p = X0.shape
    m = Y0.shape[1]
    XL, YL = np.zeros((p, ncomp)), np.zeros((m, ncomp))
    XS, YS = np.zeros((n, ncomp)), np.zeros((n, ncomp))
    W, V = np.zeros((p, ncomp)), np.zeros((p, ncomp))
    Cov = X0.T @ Y0
    for i in range(ncomp):
        U, s, Vt = np.linalg.svd(Cov, full_matrices=False)
        r, c, sv = U[:, 0], Vt[0], s[0]
        t = X0 @ r
        nt = np.linalg.norm(t)
        if nt == 0:
            break
        t = t / nt
        XL[:, i] = X0.T @ t
        q = sv * c / nt
        YL[:, i] = q
        XS[:, i] = t
        YS[:, i] = Y0 @ q
        W[:, i] = r / nt
        v = XL[:, i].copy()
        for _ in range(2):
            v -= V[:, :i] @ (V[:, :i].T @ v)
        v /= np.linalg.norm(v)
        V[:, i] = v
        Cov = Cov - np.outer(v, v @ Cov)
        Cov = Cov - V[:, :i + 1] @ (V[:, :i + 1].T @ Cov)
    for i in range(ncomp):
        for _ in range(2):
            YS[:, i] -= XS[:, :i] @ (XS[:, :i].T @ YS[:, i])
    return XL, YL, XS, YS, W


@builtin("plsregress")
def _plsregress(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("plsregress: expected plsregress(X, Y, ncomp, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["CV", "MCReps", "Options"], "plsregress")
    if len(pos) > 3:
        raise MatlabError("Too many input arguments for plsregress.")
    X = np.asarray(fnum(pos[0]), dtype=float)
    Y = np.asarray(fnum(pos[1]), dtype=float)
    if Y.ndim == 2 and Y.shape[0] == 1 and X.shape[0] != 1:
        Y = Y.T
    n, p = X.shape
    if Y.shape[0] != n:
        raise MatlabError("plsregress: X and Y must have the same number of rows.")
    if np.isnan(X).any() or np.isnan(Y).any():
        raise MatlabError("plsregress: missing values (NaN) are not supported.")
    maxc = min(n - 1, p)
    ncomp = intval(pos[2], "ncomp") if len(pos) > 2 and not is_empty_arg(pos[2]) else maxc
    if not 1 <= ncomp <= maxc:
        raise MatlabError(f"plsregress: ncomp must be between 1 and {maxc}.")
    if "mcreps" in opts and intval(opts["mcreps"], "MCReps") != 1:
        raise MatlabError("plsregress: MCReps other than 1 is not supported.")
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - mx, Y - my
    XL, YL, XS, YS, W = _simpls(X0, Y0, ncomp)
    beta = W @ YL.T
    BETA = np.vstack([my - mx @ beta, beta])
    with np.errstate(all="ignore"):
        pctvar = np.vstack([np.sum(XL ** 2, axis=0) / np.sum(X0 ** 2), np.sum(YL ** 2, axis=0) / np.sum(Y0 ** 2)])

    def sse_path(Xtr, Ytr, Xte, Yte):
        mxt, myt = Xtr.mean(axis=0), Ytr.mean(axis=0)
        _, YLt, _, _, Wt = _simpls(Xtr - mxt, Ytr - myt, ncomp)
        sx = np.empty(ncomp + 1)
        sy = np.empty(ncomp + 1)
        Xc, Yc = Xte - mxt, Yte - myt
        XLt = (Xtr - mxt).T @ ((Xtr - mxt) @ Wt)
        for k in range(ncomp + 1):
            sx[k] = np.sum((Xc - (Xc @ Wt[:, :k]) @ XLt[:, :k].T) ** 2)
            sy[k] = np.sum((Yc - Xc @ (Wt[:, :k] @ YLt[:, :k].T)) ** 2)
        return sx, sy

    folds = _cv_folds(interp, opts, n, "plsregress")
    if folds is None:
        sx, sy = sse_path(X, Y, X, Y)
        mse = np.vstack([sx, sy]) / n
    else:
        tot = np.zeros((2, ncomp + 1))
        for tr, te in folds:
            sx, sy = sse_path(X[tr], Y[tr], X[te], Y[te])
            tot += np.vstack([sx, sy])
        mse = tot / n
    Xres = X0 - XS @ XL.T
    Yres = Y0 - XS @ YL.T
    with np.errstate(all="ignore"):
        t2 = np.sum(XS ** 2 / np.var(XS, axis=0, ddof=1), axis=1)
    stats = struct(W=W, T2=t2.reshape(-1, 1), Xresiduals=Xres, Yresiduals=Yres)
    return outs(nargout, XL, YL, XS, YS, BETA, pctvar, mse, stats)


# -- multinomial regression ----------------------------------------------------------------------------------

def _mn_response(Y: Any, n: int, fname: str) -> np.ndarray:
    """n-by-k matrix of counts from category numbers, a categorical array or a count matrix."""
    if is_categorical(Y) or text_list(Y) is not None:
        lab = Labels(Y)
        if lab.n != n:
            raise MatlabError(f"{fname}: Y must have one value per row of X.")
        M = np.zeros((n, lab.k))
        ok = lab.codes >= 0
        M[np.flatnonzero(ok), lab.codes[ok]] = 1
        M[~ok] = np.nan
        return M
    A = np.asarray(fnum(Y), dtype=float)
    if A.ndim == 2 and A.shape[1] > 1 and A.shape[0] == n:
        return A
    v = A.flatten(order="F")
    if v.size != n:
        raise MatlabError(f"{fname}: Y must have one value per row of X.")
    good = ~np.isnan(v)
    if np.any(v[good] < 1) or np.any(v[good] != np.round(v[good])):
        raise MatlabError(f"{fname}: category numbers must be positive integers.")
    k = int(v[good].max())
    M = np.zeros((n, k))
    M[np.flatnonzero(good), v[good].astype(int) - 1] = 1
    M[~good] = np.nan
    return M


def _mn_probs(model: str, link: Link, B: np.ndarray, X: np.ndarray, k: int, parallel: bool) -> np.ndarray:
    n, p = X.shape
    if model == "nominal":
        eta = np.column_stack([B[0, j] + X @ B[1:, j] for j in range(k - 1)] + [np.zeros(n)])
        eta -= eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(axis=1, keepdims=True)
    if parallel:
        theta, beta = B[:k - 1, 0], B[k - 1:, 0]
        eta = theta[None, :] + (X @ beta)[:, None]
    else:
        eta = B[0][None, :] + X @ B[1:]
    g = link.inv(eta)
    if model == "ordinal":
        cum = np.column_stack([np.zeros(n), g, np.ones(n)])
        return np.diff(cum, axis=1)
    pi = np.empty((n, k))            # hierarchical: g is P(Y = j | Y >= j)
    rem = np.ones(n)
    for j in range(k - 1):
        pi[:, j] = rem * g[:, j]
        rem = rem * (1 - g[:, j])
    pi[:, k - 1] = rem
    return pi


def _mn_options(opts: dict, fname: str):
    model = choice(opts["model"], ["nominal", "ordinal", "hierarchical"], "model") if "model" in opts else "nominal"
    link = Link(opts.get("link", "logit"), "binomial")
    if model == "nominal" and link.name != "logit":
        raise MatlabError(f"{fname}: the nominal model uses the logit link.")
    parallel = model == "ordinal"
    if "interactions" in opts:
        parallel = not as_bool(opts["interactions"], "interactions")
    if model == "nominal" and parallel:
        raise MatlabError(f"{fname}: a nominal model needs 'interactions', 'on'.")
    return model, link, parallel


@builtin("mnrfit")
def _mnrfit(interp, args, nargout):
    import scipy.stats as st
    from scipy.optimize import minimize
    if len(args) < 2:
        raise MatlabError("mnrfit: expected mnrfit(X, Y, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["model", "interactions", "link", "estdisp"], "mnrfit")
    if len(pos) != 2:
        raise MatlabError("mnrfit: expected mnrfit(X, Y, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError("mnrfit: X must be a matrix.")
    model, link, parallel = _mn_options(opts, "mnrfit")
    estdisp = as_bool(opts["estdisp"], "estdisp") if "estdisp" in opts else False
    M = _mn_response(pos[1], X.shape[0], "mnrfit")
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(M).any(axis=1))
    X, M = X[ok], M[ok]
    n, p = X.shape
    k = M.shape[1]
    if k < 2:
        raise MatlabError("mnrfit: the response needs at least two categories.")
    m = M.sum(axis=1)
    if parallel:
        shape = (k - 1 + p, 1)
        cum = np.cumsum(M.sum(axis=0))[:-1] / M.sum()
        start = np.concatenate([link.g(np.clip(cum, 1e-6, 1 - 1e-6)), np.zeros(p)])
    else:
        shape = (p + 1, k - 1)
        start = np.zeros(shape)
        tot = M.sum(axis=0) / M.sum()
        if model == "nominal":
            start[0] = np.log(np.clip(tot[:-1], 1e-9, None) / max(tot[-1], 1e-9))
        elif model == "ordinal":
            start[0] = link.g(np.clip(np.cumsum(tot)[:-1], 1e-6, 1 - 1e-6))
        else:
            rem = 1 - np.concatenate([[0], np.cumsum(tot)[:-2]])
            start[0] = link.g(np.clip(tot[:-1] / rem, 1e-6, 1 - 1e-6))
        start = start.flatten(order="F")

    def nll(b):
        P = _mn_probs(model, link, b.reshape(shape, order="F"), X, k, parallel)
        if np.any(P <= 0) or not np.all(np.isfinite(P)):
            return 1e300
        return -float(np.sum(M * np.log(P)))

    sol = minimize(nll, start, method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
    b = sol.x
    from ._stats_util import num_hessian
    for _ in range(25):                       # Newton polishing with a numerical Hessian
        H = num_hessian(nll, b, 1e-4)
        g = np.array([(nll(b + e) - nll(b - e)) / (2 * np.linalg.norm(e)) for e in np.eye(b.size) * 1e-6])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)) or nll(b - step) > nll(b) + 1e-12:
            break
        b = b - step
        if np.max(np.abs(step)) < 1e-10:
            break
    B = b.reshape(shape, order="F")
    if nargout <= 1:
        return B
    P = _mn_probs(model, link, B, X, k, parallel)
    with np.errstate(all="ignore"):
        dev = 2 * float(np.sum(np.where(M > 0, M * np.log(M / (m[:, None] * P)), 0.0)))
        H = num_hessian(nll, b, 1e-4)
        cov = np.linalg.pinv(H)
        dfe = n * (k - 1) - b.size
        pearson = (M - m[:, None] * P) / np.sqrt(m[:, None] * P)
        sfit = float(np.sqrt(np.sum(pearson ** 2) / dfe)) if dfe > 0 else np.nan
        s = sfit if estdisp else 1.0
        cov = cov * s ** 2
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        t = b / se
        pv = 2 * (st.t.sf(np.abs(t), dfe) if estdisp else st.norm.sf(np.abs(t)))
        corr = cov / np.outer(se, se)
        residd = np.sign(M - m[:, None] * P) * np.sqrt(2 * np.abs(np.where(M > 0, M * np.log(M / (m[:, None] * P)),
                0.0)))
    stats = struct(beta=B, dfe=float(dfe), sfit=sfit, s=float(s), estdisp=estdisp, covb=cov, coeffcorr=corr,
                   se=se.reshape(shape, order="F"), t=t.reshape(shape, order="F"), p=pv.reshape(shape, order="F"),
                   resid=M - m[:, None] * P, residp=pearson, residd=residd)
    return outs(nargout, B, dev, stats)


@builtin("mnrval")
def _mnrval(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("mnrval: expected mnrval(B, X, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["model", "interactions", "link", "type", "confidence"], "mnrval")
    if len(pos) > 3 or nargout > 1 or "confidence" in opts:
        raise MatlabError("mnrval: confidence bounds (the stats input and extra outputs) are not supported.")
    if len(pos) == 3 and not isinstance(pos[2], Struct) and not is_empty_arg(pos[2]):
        raise MatlabError("mnrval: sample sizes as a third input are not supported.")
    B = np.asarray(fnum(pos[0]), dtype=float)
    X = np.asarray(fnum(pos[1]), dtype=float)
    model, link, parallel = _mn_options(opts, "mnrval")
    kind = choice(opts["type"], ["category", "cumulative", "conditional"], "type") if "type" in opts else "category"
    if X.ndim != 2:
        raise MatlabError("mnrval: X must be a matrix.")
    p = X.shape[1]
    if parallel:
        if B.shape[1] != 1 and B.shape[0] == 1:
            B = B.T
        k = B.shape[0] - p + 1
        if k < 2:
            raise MatlabError("mnrval: B does not match the number of predictors.")
    else:
        if B.shape[0] != p + 1:
            raise MatlabError("mnrval: B must have one row per predictor plus one for the intercepts.")
        k = B.shape[1] + 1
    P = _mn_probs(model, link, B, X, k, parallel)
    if kind == "category":
        return P
    if kind == "cumulative":
        return np.cumsum(P, axis=1)[:, :-1]
    rem = 1 - np.column_stack([np.zeros(P.shape[0]), np.cumsum(P, axis=1)[:, :-2]])
    with np.errstate(all="ignore"):
        return P[:, :-1] / rem
