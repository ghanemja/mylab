"""More learners: support vector machines, linear models for wide data, error-correcting output codes,
tree ensembles (boosting, bagging, TreeBagger), Gaussian process regression and small neural networks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, char
from ._registry import builtin
from ._stats_util import (
    Labels, StatObject, as_bool, cellstr, choice, get_opts, intval, outs, rng_of, scalar, struct, vec,
)
from ._util import fnum, is_text, text
from .stats_ml import (
    COMMON_OPTS, Data, DiscriminantModel, KNNModel, Model, NaiveBayesModel, Tree, TreeModel, _KNN_OPTS, _TREE_OPTS,
    class_prior, finish, grow_tree, learner_spec, prepare, prior_weights, tree_params,
)

# -- sequential minimal optimisation -----------------------------------------------------------------------

def smo(Q: np.ndarray, p: np.ndarray, y: np.ndarray, C: np.ndarray, tol: float = 1e-4, max_iter: int = 0):
    """Minimise 0.5 a'Qa + p'a subject to y'a = 0 and 0 <= a <= C. Returns (a, bias, iterations, converged)."""
    n = p.size
    a = np.zeros(n)
    G = p.astype(float).copy()
    qd = np.diag(Q).copy()
    tau = 1e-12
    max_iter = max_iter or max(10_000_000 // max(n, 1), 100 * n, 10000)
    converged = False
    it = 0
    gmax = gmin = 0.0
    while it < max_iter:
        it += 1
        up = ((y > 0) & (a < C)) | ((y < 0) & (a > 0))
        low = ((y > 0) & (a > 0)) | ((y < 0) & (a < C))
        yg = -y * G
        if not up.any() or not low.any():
            converged = True
            break
        i = int(np.argmax(np.where(up, yg, -np.inf)))
        gmax = yg[i]
        gmin = float(np.min(np.where(low, yg, np.inf)))
        if gmax - gmin < tol:
            converged = True
            break
        Qi = Q[i]
        b = gmax - yg
        cand = low & (b > 0)
        quad = qd[i] + qd - 2 * y[i] * y * Qi
        quad = np.where(quad > 0, quad, tau)
        j = int(np.argmin(np.where(cand, -(b * b) / quad, np.inf)))
        Qj = Q[j]
        ai, aj = a[i], a[j]
        if y[i] != y[j]:
            q = max(qd[i] + qd[j] + 2 * Qi[j], tau)
            delta = (-G[i] - G[j]) / q
            diff = ai - aj
            ni, nj = ai + delta, aj + delta
            if diff > 0:
                if nj < 0:
                    nj, ni = 0.0, diff
            elif ni < 0:
                ni, nj = 0.0, -diff
            if diff > C[i] - C[j]:
                if ni > C[i]:
                    ni, nj = C[i], C[i] - diff
            elif nj > C[j]:
                nj, ni = C[j], C[j] + diff
        else:
            q = max(qd[i] + qd[j] - 2 * Qi[j], tau)
            delta = (G[i] - G[j]) / q
            s = ai + aj
            ni, nj = ai - delta, aj + delta
            if s > C[i]:
                if ni > C[i]:
                    ni, nj = C[i], s - C[i]
            elif nj < 0:
                nj, ni = 0.0, s
            if s > C[j]:
                if nj > C[j]:
                    nj, ni = C[j], s - C[j]
            elif ni < 0:
                ni, nj = 0.0, s
        G += Qi * (ni - ai) + Qj * (nj - aj)
        a[i], a[j] = ni, nj
    free = (a > 1e-12 * np.maximum(C, 1e-300)) & (a < C * (1 - 1e-12))
    yg = -y * G
    bias = float(np.mean(yg[free])) if free.any() else float((gmax + gmin) / 2)
    return a, bias, it, converged


def kernel_matrix(A: np.ndarray, B: np.ndarray, kind: str, order: int = 3, offset: float = 0.0) -> np.ndarray:
    if kind == "linear":
        K = A @ B.T
    elif kind == "polynomial":
        K = (1 + A @ B.T) ** order
    else:
        d2 = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T
        K = np.exp(-np.clip(d2, 0, None))
    return K + offset


_SVM_OPTS = ["KernelFunction", "KernelScale", "BoxConstraint", "Standardize", "PolynomialOrder", "KernelOffset",
             "Solver", "Nu", "OutlierFraction", "Alpha", "CacheSize", "ClipAlphas", "DeltaGradientTolerance",
             "GapTolerance", "KKTTolerance", "IterationLimit", "ShrinkagePeriod", "Verbose", "RemoveDuplicates",
             "SaveSupportVectors", "Epsilon", "NumPrint"]


class SVMBase(Model):
    def setup_kernel(self, default_kernel: str = "linear") -> None:
        d, o = self.d, self.d.opts
        for bad in ("alpha", "nu"):
            if bad in o:
                raise MatlabError(f"{self.fname}: the option '{bad}' is not supported.")
        if "outlierfraction" in o and scalar(o["outlierfraction"], "OutlierFraction") != 0:
            raise MatlabError(f"{self.fname}: the option 'OutlierFraction' is not supported.")
        if "solver" in o:
            choice(o["solver"], ["SMO", "ISDA", "L1QP"], "Solver")     # every solver reaches the same optimum
        kf = text(o["kernelfunction"]).lower() if "kernelfunction" in o else default_kernel
        kf = {"rbf": "gaussian"}.get(kf, kf)
        if kf not in ("linear", "gaussian", "polynomial"):
            raise MatlabError(f"{self.fname}: KernelFunction must be 'linear', 'gaussian' (or 'rbf') or 'polynomial'.")
        self.kernel = kf
        self.order = intval(o["polynomialorder"], "PolynomialOrder") if "polynomialorder" in o else 3
        if "polynomialorder" in o and kf != "polynomial":
            raise MatlabError(f"{self.fname}: PolynomialOrder applies to the polynomial kernel only.")
        self.offset = scalar(o["kerneloffset"], "KernelOffset") if "kerneloffset" in o else 0.0
        self.standardize = as_bool(o["standardize"], "Standardize") if "standardize" in o else False
        w = self.wn
        self.mu = w @ d.X
        sd = np.sqrt(np.sum(w[:, None] * (d.X - self.mu) ** 2, axis=0) / max(1 - np.sum(w ** 2), 1e-300))
        self.sigma = np.where(sd > 0, sd, 1.0)
        Z = (d.X - self.mu) / self.sigma if self.standardize else d.X
        ks = o.get("kernelscale")
        if ks is None:
            self.scale = 1.0
        elif is_text(ks):
            if text(ks).lower() != "auto":
                raise MatlabError(f"{self.fname}: KernelScale must be a positive number or 'auto'.")
            sub = Z[:: max(1, Z.shape[0] // 500)]
            d2 = np.sum(sub ** 2, axis=1)[:, None] + np.sum(sub ** 2, axis=1)[None, :] - 2 * sub @ sub.T
            med = np.median(d2[np.triu_indices(sub.shape[0], 1)]) if sub.shape[0] > 1 else 1.0
            self.scale = float(np.sqrt(med)) if med > 0 else 1.0
        else:
            self.scale = scalar(ks, "KernelScale")
            if self.scale <= 0:
                raise MatlabError(f"{self.fname}: KernelScale must be positive.")
        self.Z = Z / self.scale
        self.tol = 1e-4
        if "deltagradienttolerance" in o and scalar(o["deltagradienttolerance"], "DeltaGradientTolerance") > 0:
            self.tol = scalar(o["deltagradienttolerance"], "DeltaGradientTolerance")
        self.iter_limit = intval(o["iterationlimit"], "IterationLimit") if "iterationlimit" in o else 0

    def zmap(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sigma if self.standardize else X) / self.scale

    def kernel_with(self, X: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return kernel_matrix(self.zmap(X), self.Z[rows], self.kernel, self.order, self.offset)

    def kernel_props(self) -> Any:
        return struct(Function=self.kernel, Scale=float(self.scale), **(
            {"Order": float(self.order)} if self.kernel == "polynomial" else {}))


class SVMClassifier(SVMBase):
    matlab_class = "ClassificationSVM"
    extra_props = ("Alpha", "Beta", "Bias", "BoxConstraints", "SupportVectors", "SupportVectorLabels",
                   "IsSupportVector", "KernelParameters", "Mu", "Sigma", "ConvergenceInfo")

    def __init__(self, interp, d: Data, fname: str = "fitcsvm"):
        super().__init__(d, fname)
        self.interp = interp
        if d.labels.k != 2:
            raise MatlabError(f"{fname}: the response has {d.labels.k} classes; fitcsvm trains two-class models "
                              "(use fitcecoc for more classes).")
        present = np.unique(d.codes)
        if present.size < 2:
            raise MatlabError(f"{fname}: one-class learning is not supported; both classes need observations.")
        self.setup_kernel()
        box = scalar(d.opts["boxconstraint"], "BoxConstraint") if "boxconstraint" in d.opts else 1.0
        if box <= 0:
            raise MatlabError(f"{fname}: BoxConstraint must be positive.")
        y = np.where(d.codes == 1, 1.0, -1.0)
        self.C = box * d.n * self.wn
        K = kernel_matrix(self.Z, self.Z, self.kernel, self.order, self.offset)
        a, bias, it, conv = smo(K * np.outer(y, y), -np.ones(d.n), y, self.C, self.tol, self.iter_limit)
        self.y = y
        self.alpha, self.bias = a, bias
        self.sv = a > 1e-10 * max(box, 1e-300)
        self.info = (it, conv)

    def refit(self, d: Data) -> "SVMClassifier":
        return SVMClassifier(self.interp, d, self.fname)

    def decision(self, X: np.ndarray) -> np.ndarray:
        rows = np.flatnonzero(self.sv)
        f = self.kernel_with(X, rows) @ (self.alpha[rows] * self.y[rows]) + self.bias if rows.size else \
            np.full(X.shape[0], self.bias)
        return np.where(np.isnan(X).any(axis=1), np.nan, f)

    def scores(self, X: np.ndarray) -> np.ndarray:
        f = self.decision(X)
        return np.column_stack([-f, f])

    def get_field(self, interp, name: str) -> Any:
        rows = np.flatnonzero(self.sv)
        if name == "Alpha":
            return self.alpha[rows].reshape(-1, 1)
        if name == "Bias":
            return float(self.bias)
        if name == "Beta":
            if self.kernel != "linear":
                return np.zeros((0, 0))
            return (self.Z[rows].T @ (self.alpha[rows] * self.y[rows])).reshape(-1, 1)
        if name == "BoxConstraints":
            return self.C.reshape(-1, 1)
        if name == "SupportVectors":
            return self.Z[rows] * self.scale
        if name == "SupportVectorLabels":
            return self.y[rows].reshape(-1, 1)
        if name == "IsSupportVector":
            return self.sv.reshape(-1, 1)
        if name == "KernelParameters":
            return self.kernel_props()
        if name in ("Mu", "Sigma"):
            v = self.mu if name == "Mu" else self.sigma
            return v.reshape(1, -1) if self.standardize else np.zeros((0, 0))
        if name == "ConvergenceInfo":
            return struct(Converged=bool(self.info[1]), NumIterations=float(self.info[0]))
        return super().get_field(interp, name)


@builtin("fitcsvm")
def _fitcsvm(interp, args, nargout):
    return finish(interp, SVMClassifier(interp, prepare(interp, args, "fitcsvm", _SVM_OPTS, True)))


class SVMRegressor(SVMBase):
    matlab_class = "RegressionSVM"
    classification = False
    default_loss = "mse"
    extra_props = ("Alpha", "Beta", "Bias", "Epsilon", "BoxConstraints", "SupportVectors", "IsSupportVector",
                   "KernelParameters", "Mu", "Sigma", "ConvergenceInfo")

    def __init__(self, interp, d: Data, fname: str = "fitrsvm"):
        super().__init__(d, fname)
        self.interp = interp
        self.setup_kernel()
        o = d.opts
        q75, q25 = np.percentile(d.y, [75, 25], method="hazen")
        iqr = float(q75 - q25)
        self.epsilon = scalar(o["epsilon"], "Epsilon") if "epsilon" in o else (iqr / 13.49 if iqr > 0 else 0.1)
        dbox = iqr / 1.349 if self.kernel == "gaussian" and iqr > 0 else 1.0
        box = scalar(o["boxconstraint"], "BoxConstraint") if "boxconstraint" in o else dbox
        if box <= 0 or self.epsilon < 0:
            raise MatlabError(f"{fname}: BoxConstraint must be positive and Epsilon must not be negative.")
        n = d.n
        K = kernel_matrix(self.Z, self.Z, self.kernel, self.order, self.offset)
        Q = np.block([[K, -K], [-K, K]])
        p = np.concatenate([self.epsilon - d.y, self.epsilon + d.y])
        y2 = np.concatenate([np.ones(n), -np.ones(n)])
        self.C = box * n * self.wn
        a, bias, it, conv = smo(Q, p, y2, np.concatenate([self.C, self.C]), self.tol, self.iter_limit)
        self.coef = a[:n] - a[n:]
        self.bias = bias
        self.sv = np.abs(self.coef) > 1e-10 * max(box, 1e-300)
        self.info = (it, conv)

    def refit(self, d: Data) -> "SVMRegressor":
        return SVMRegressor(self.interp, d, self.fname)

    def response(self, X: np.ndarray) -> np.ndarray:
        rows = np.flatnonzero(self.sv)
        f = self.kernel_with(X, rows) @ self.coef[rows] + self.bias if rows.size else np.full(X.shape[0], self.bias)
        return np.where(np.isnan(X).any(axis=1), np.nan, f)

    def get_field(self, interp, name: str) -> Any:
        rows = np.flatnonzero(self.sv)
        if name == "Alpha":
            return self.coef[rows].reshape(-1, 1)
        if name == "Bias":
            return float(self.bias)
        if name == "Epsilon":
            return float(self.epsilon)
        if name == "Beta":
            return (self.Z[rows].T @ self.coef[rows]).reshape(-1, 1) if self.kernel == "linear" else np.zeros((0, 0))
        if name == "BoxConstraints":
            return self.C.reshape(-1, 1)
        if name == "SupportVectors":
            return self.Z[rows] * self.scale
        if name == "IsSupportVector":
            return self.sv.reshape(-1, 1)
        if name == "KernelParameters":
            return self.kernel_props()
        if name in ("Mu", "Sigma"):
            v = self.mu if name == "Mu" else self.sigma
            return v.reshape(1, -1) if self.standardize else np.zeros((0, 0))
        if name == "ConvergenceInfo":
            return struct(Converged=bool(self.info[1]), NumIterations=float(self.info[0]))
        return super().get_field(interp, name)


@builtin("fitrsvm")
def _fitrsvm(interp, args, nargout):
    return finish(interp, SVMRegressor(interp, prepare(interp, args, "fitrsvm", _SVM_OPTS, False)))


# -- linear models (fitclinear / fitrlinear) ----------------------------------------------------------------------

_LIN_OPTS = ["Learner", "Regularization", "Lambda", "Solver", "FitBias", "Epsilon", "ObservationsIn", "Beta",
             "Bias", "BetaTolerance", "GradientTolerance", "IterationLimit", "Verbose", "PostFitBias", "BatchSize",
             "LearnRate", "PassLimit", "OptimizeLearnRate", "TruncationPeriod", "NumCheckConvergence"]


def _linear_fit(X, y, w, learner: str, reg: str, lam: float, fit_bias: bool, eps: float):
    """Minimise sum_i w_i loss_i + lam * penalty for one lambda; returns (beta, bias)."""
    from scipy.optimize import linprog, minimize
    n, p = X.shape
    if learner == "leastsquares" and reg == "ridge":
        mx = w @ X if fit_bias else np.zeros(p)
        my = float(w @ y) if fit_bias else 0.0
        Xc = X - mx
        A = (Xc * w[:, None]).T @ Xc + lam * np.eye(p)
        beta = np.linalg.solve(A, (Xc * w[:, None]).T @ (y - my))
        return beta, my - float(mx @ beta)
    if learner == "leastsquares":
        from .stats_regress import _cd_path
        mx = w @ X if fit_bias else np.zeros(p)
        my = float(w @ y) if fit_bias else 0.0
        beta = _cd_path(X - mx, y - my, w, np.array([lam]), 1.0, 1e-9, 100000)[:, 0]
        return beta, my - float(mx @ beta)
    if learner == "logistic":
        def obj(t):
            f = X @ t[:p] + (t[p] if fit_bias else 0.0)
            m = y * f
            loss = np.sum(w * np.logaddexp(0, -m))
            g = -(w * y) / (1 + np.exp(m))
            grad = np.concatenate([X.T @ g, [g.sum()] if fit_bias else []])
            if reg == "ridge":
                loss += 0.5 * lam * np.sum(t[:p] ** 2)
                grad[:p] += lam * t[:p]
            return loss, grad
        size = p + (1 if fit_bias else 0)
        if reg == "ridge":
            sol = minimize(obj, np.zeros(size), jac=True, method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-15,
                                                                                    "gtol": 1e-10})
            t = sol.x
        else:                                    # proximal gradient with backtracking for the L1 penalty
            t = np.zeros(size)
            step = 1.0
            f0, g0 = obj(t)
            for _ in range(20000):
                while True:
                    cand = t - step * g0
                    cand[:p] = np.sign(cand[:p]) * np.maximum(np.abs(cand[:p]) - step * lam, 0)
                    f1, g1 = obj(cand)
                    dlt = cand - t
                    if f1 <= f0 + g0 @ dlt + np.sum(dlt ** 2) / (2 * step) + 1e-15:
                        break
                    step /= 2
                done = np.max(np.abs(dlt)) < 1e-10
                t, f0, g0 = cand, f1, g1
                step *= 1.5
                if done:
                    break
        return t[:p], float(t[p]) if fit_bias else 0.0
    # hinge or epsilon-insensitive loss: solved exactly as a linear or quadratic program
    classification = eps < 0
    if reg == "ridge" and fit_bias:
        if classification:
            a, bias, _, _ = smo((X @ X.T) * np.outer(y, y), -np.ones(n), y, w / lam, 1e-6)
            return X.T @ (a * y), bias
        K = X @ X.T
        Q = np.block([[K, -K], [-K, K]])
        a, bias, _, _ = smo(Q, np.concatenate([eps - y, eps + y]), np.concatenate([np.ones(n), -np.ones(n)]),
                            np.concatenate([w, w]) / lam, 1e-6)
        return X.T @ (a[:n] - a[n:]), bias
    if reg == "ridge":
        def obj2(t):
            f = X @ t
            if classification:
                viol = np.maximum(0, 1 - y * f)
                return np.sum(w * viol) + 0.5 * lam * t @ t
            return np.sum(w * np.maximum(0, np.abs(y - f) - eps)) + 0.5 * lam * t @ t
        sol = minimize(obj2, np.zeros(p), method="Powell", options={"xtol": 1e-10, "ftol": 1e-12, "maxiter": 20000})
        return sol.x, 0.0
    # lasso: variables [beta+, beta-, bias+, bias-, slack]
    nb = 2 if fit_bias else 0
    c = np.concatenate([np.full(2 * p, lam), np.zeros(nb), w])
    bcols = np.column_stack([np.ones(n), -np.ones(n)]) if fit_bias else np.zeros((n, 0))
    if classification:
        Yx = X * y[:, None]
        A = np.hstack([-Yx, Yx, -bcols * y[:, None], -np.eye(n)])
        b = -np.ones(n)
    else:
        A = np.vstack([np.hstack([X, -X, bcols, -np.eye(n)]), np.hstack([-X, X, -bcols, -np.eye(n)])])
        b = np.concatenate([y + eps, eps - y])
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * c.size, method="highs")
    if res.status != 0:
        raise MatlabError("The linear program for the L1-penalised fit did not converge.")
    z = res.x
    return z[:p] - z[p:2 * p], float(z[2 * p] - z[2 * p + 1]) if fit_bias else 0.0


def _linear_common(d: Data, fname: str, classification: bool):
    o = d.opts
    learners = ["svm", "logistic"] if classification else ["svm", "leastsquares"]
    learner = choice(o["learner"], learners, "Learner") if "learner" in o else "svm"
    reg = choice(o["regularization"], ["ridge", "lasso"], "Regularization") if "regularization" in o else "ridge"
    lam_v = o.get("lambda")
    if lam_v is None or (is_text(lam_v) and text(lam_v).lower() == "auto"):
        lams = np.array([1.0 / d.n])
    else:
        lams = vec(lam_v)
        if np.any(lams < 0):
            raise MatlabError(f"{fname}: Lambda must not be negative.")
    fit_bias = as_bool(o["fitbias"], "FitBias") if "fitbias" in o else True
    for bad in ("beta", "bias"):
        if bad in o:
            raise MatlabError(f"{fname}: initial values ('{bad}') are not supported.")
    return learner, reg, lams, fit_bias


class LinearClassifier(Model):
    matlab_class = "ClassificationLinear"
    extra_props = ("Beta", "Bias", "Lambda", "Learner", "Regularization")

    def __init__(self, interp, d: Data, fname: str = "fitclinear"):
        super().__init__(d, fname)
        self.interp = interp
        if d.labels.k != 2:
            raise MatlabError(f"{fname}: fitclinear trains two-class models (use fitcecoc for more classes).")
        self.learner, self.reg, self.lams, fit_bias = _linear_common(d, fname, True)
        y = np.where(d.codes == 1, 1.0, -1.0)
        fits = [_linear_fit(d.X, y, self.wn, self.learner, self.reg, max(float(lm), 1e-12), fit_bias, -1.0)
                for lm in self.lams]
        self.beta = np.column_stack([f[0] for f in fits])
        self.bias = np.array([f[1] for f in fits])
        if self.learner == "logistic":
            self.transform = "logit"

    def refit(self, d: Data) -> "LinearClassifier":
        return LinearClassifier(self.interp, d, self.fname)

    def scores(self, X: np.ndarray) -> np.ndarray:
        f = X @ self.beta[:, 0] + self.bias[0]
        return np.column_stack([-f, f])

    def get_field(self, interp, name: str) -> Any:
        if name == "Beta":
            return self.beta
        if name == "Bias":
            return self.bias.reshape(1, -1)
        if name == "Lambda":
            return self.lams.reshape(1, -1)
        if name == "Learner":
            return self.learner
        if name == "Regularization":
            return "ridge (L2)" if self.reg == "ridge" else "lasso (L1)"
        return super().get_field(interp, name)


class LinearRegressor(Model):
    matlab_class = "RegressionLinear"
    classification = False
    default_loss = "mse"
    extra_props = ("Beta", "Bias", "Lambda", "Learner", "Regularization", "Epsilon")

    def __init__(self, interp, d: Data, fname: str = "fitrlinear"):
        super().__init__(d, fname)
        self.interp = interp
        self.learner, self.reg, self.lams, fit_bias = _linear_common(d, fname, False)
        q75, q25 = np.percentile(d.y, [75, 25], method="hazen")
        self.epsilon = scalar(d.opts["epsilon"], "Epsilon") if "epsilon" in d.opts else float(q75 - q25) / 13.49
        fits = [_linear_fit(d.X, d.y, self.wn, self.learner, self.reg, max(float(lm), 1e-12), fit_bias,
                            0.0 if self.learner == "leastsquares" else self.epsilon) for lm in self.lams]
        self.beta = np.column_stack([f[0] for f in fits])
        self.bias = np.array([f[1] for f in fits])

    def refit(self, d: Data) -> "LinearRegressor":
        return LinearRegressor(self.interp, d, self.fname)

    def response(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta[:, 0] + self.bias[0]

    def m_predict(self, interp, args, nargout):
        X = self.d.encode_x(args[0], "predict")
        return X @ self.beta + self.bias[None, :]

    get_field = LinearClassifier.get_field


def _obs_in_columns(args: list) -> list:
    """Honour 'ObservationsIn', 'columns' by transposing X."""
    out = list(args)
    for i in range(2, len(out) - 1):
        if is_text(out[i]) and text(out[i]).lower() == "observationsin":
            if choice(out[i + 1], ["rows", "columns"], "ObservationsIn") == "columns":
                out[0] = np.asarray(fnum(out[0])).T
            del out[i:i + 2]
            break
    return out


@builtin("fitclinear")
def _fitclinear(interp, args, nargout):
    return finish(interp, LinearClassifier(interp, prepare(interp, _obs_in_columns(args), "fitclinear", _LIN_OPTS,
            True)))


@builtin("fitrlinear")
def _fitrlinear(interp, args, nargout):
    return finish(interp, LinearRegressor(interp, prepare(interp, _obs_in_columns(args), "fitrlinear", _LIN_OPTS,
            False)))


# -- error-correcting output codes ------------------------------------------------------------------------------------

def _binary_learner(interp, kind: str, largs: list, d: Data):
    fname = "fitcecoc"
    opts_names = {"svm": _SVM_OPTS, "linear": _LIN_OPTS, "knn": _KNN_OPTS, "tree": _TREE_OPTS,
                  "discriminant": ["DiscrimType", "Gamma", "Delta"],
                  "naivebayes": ["DistributionNames", "Kernel", "Support", "Width"]}[kind]
    _, o = get_opts(largs, 0, COMMON_OPTS + opts_names, fname)
    d.opts = o
    if kind == "svm":
        return SVMClassifier(interp, d, fname)
    if kind == "linear":
        return LinearClassifier(interp, d, fname)
    if kind == "knn":
        return KNNModel(interp, d, fname)
    if kind == "tree":
        return TreeModel(interp, d, fname, True)
    if kind == "discriminant":
        return DiscriminantModel(interp, d, fname)
    return NaiveBayesModel(interp, d, fname)


def _coding(name: Any, K: int) -> tuple[np.ndarray, str]:
    if not is_text(name):
        M = np.asarray(fnum(name), dtype=float)
        if M.ndim != 2 or M.shape[0] != K or not np.all(np.isin(M, (-1, 0, 1))):
            raise MatlabError("fitcecoc: a coding matrix needs one row per class with entries -1, 0 and 1.")
        return M, "custom"
    kind = choice(name, ["onevsone", "allpairs", "onevsall", "ordinal", "binarycomplete", "ternarycomplete"], "Coding")
    kind = "onevsone" if kind == "allpairs" else kind
    if kind == "onevsone":
        cols = []
        for a in range(K):
            for b in range(a + 1, K):
                c = np.zeros(K)
                c[a], c[b] = 1, -1
                cols.append(c)
    elif kind == "onevsall":
        cols = [np.where(np.arange(K) == a, 1.0, -1.0) for a in range(K)]
    elif kind == "ordinal":
        cols = [np.where(np.arange(K) <= a, -1.0, 1.0) for a in range(K - 1)]
    elif kind == "binarycomplete":
        cols = []
        for mask in range(1, 2 ** (K - 1)):
            cols.append(np.array([1.0 if (mask >> a) & 1 else -1.0 for a in range(K)]))
    else:
        from itertools import product
        cols = []
        for combo in product((-1, 0, 1), repeat=K):
            c = np.array(combo, dtype=float)
            if np.any(c == 1) and np.any(c == -1) and c[np.flatnonzero(c)[0]] == 1:
                cols.append(c)
    return np.column_stack(cols), kind


_BIN_LOSS = {"hamming": lambda y, s: (1 - np.sign(y * s)) / 2, "linear": lambda y, s: (1 - y * s) / 2,
             "logit": lambda y, s: np.log1p(np.exp(-y * s)) / (2 * np.log(2)),
             "exponential": lambda y, s: np.exp(-y * s) / 2,
             "binodeviance": lambda y, s: np.log1p(np.exp(-2 * y * s)) / (2 * np.log(2)),
             "hinge": lambda y, s: np.maximum(0, 1 - y * s) / 2,
             "quadratic": lambda y, s: (1 - y * (2 * s - 1)) ** 2 / 2}


class ECOCModel(Model):
    matlab_class = "ClassificationECOC"
    extra_props = ("BinaryLearners", "CodingMatrix", "CodingName", "BinaryLoss", "LearnerWeights")

    def __init__(self, interp, d: Data, fname: str = "fitcecoc"):
        super().__init__(d, fname)
        self.interp = interp
        o = d.opts
        K = d.labels.k
        if K < 2:
            raise MatlabError("fitcecoc: the response needs at least two classes.")
        self.M, self.coding = _coding(o.get("coding", char("onevsone")), K)
        self.kind, self.largs = learner_spec(o["learners"], fname) if "learners" in o else ("svm", [])
        self.learners = []
        for col in self.M.T:
            pos, neg = np.flatnonzero(col > 0), np.flatnonzero(col < 0)
            rows = np.flatnonzero(np.isin(d.codes, np.concatenate([pos, neg])))
            sub = d.subset(rows)
            sub.labels = Labels(np.isin(d.codes[rows], pos).astype(float).reshape(-1, 1), np.array([[0.0], [1.0]]))
            sub.codes = sub.labels.codes
            sub.w = self.wn[rows]
            self.learners.append(_binary_learner(interp, self.kind, self.largs, sub))
        default = {"svm": "hinge", "linear": "hinge"}.get(self.kind, "quadratic")
        if self.kind == "linear" and self.learners and self.learners[0].learner == "logistic":
            default = "quadratic"
        self.binloss = choice(o["binaryloss"], list(_BIN_LOSS), "BinaryLoss") if "binaryloss" in o else default
        self.decoding = choice(o["decoding"], ["lossweighted", "lossbased"], "Decoding") if "decoding" in o \
            else "lossweighted"

    def refit(self, d: Data) -> "ECOCModel":
        return ECOCModel(self.interp, d, self.fname)

    def pb_scores(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for m in self.learners:
            _, St, _ = m.classify(self.interp, X)
            cols.append(St[:, 1])
        return np.column_stack(cols)

    def neg_loss(self, PB: np.ndarray, binloss: str | None = None, decoding: str | None = None) -> np.ndarray:
        g = _BIN_LOSS[binloss or self.binloss]
        K, L = self.M.shape
        out = np.empty((PB.shape[0], K))
        with np.errstate(all="ignore"):
            for k in range(K):
                m = self.M[k]
                loss = np.abs(m)[None, :] * g(m[None, :], PB)
                denom = np.sum(np.abs(m)) if (decoding or self.decoding) == "lossweighted" else L
                out[:, k] = -np.nansum(loss, axis=1) / denom
        return out

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self.neg_loss(self.pb_scores(X))

    def m_predict(self, interp, args, nargout):
        pos, opts = get_opts(args, 1, ["BinaryLoss", "Decoding", "Verbose", "ObservationsIn"], "predict")
        X = self.d.encode_x(pos[0], "predict")
        bl = choice(opts["binaryloss"], list(_BIN_LOSS), "BinaryLoss") if "binaryloss" in opts else None
        dec = choice(opts["decoding"], ["lossweighted", "lossbased"], "Decoding") if "decoding" in opts else None
        PB = self.pb_scores(X)
        NL = self.neg_loss(PB, bl, dec)
        return outs(nargout, self.d.labels.decode(np.argmax(NL, axis=1)), NL, PB)

    def get_field(self, interp, name: str) -> Any:
        if name == "BinaryLearners":
            return Cell.from_list(self.learners, column=True)
        if name == "CodingMatrix":
            return self.M
        if name == "CodingName":
            return self.coding
        if name == "BinaryLoss":
            return self.binloss
        if name == "LearnerWeights":
            return np.array([[float(m.d.n) for m in self.learners]])
        return super().get_field(interp, name)


@builtin("fitcecoc")
def _fitcecoc(interp, args, nargout):
    known = ["Coding", "Learners", "FitPosterior", "NumBins", "NumConcurrent", "ObservationsIn", "Options",
             "Verbose", "BinaryLoss", "Decoding"]
    d = prepare(interp, args, "fitcecoc", known, True)
    if "fitposterior" in d.opts and as_bool(d.opts["fitposterior"], "FitPosterior"):
        raise MatlabError("fitcecoc: the option 'FitPosterior' is not supported.")
    return finish(interp, ECOCModel(interp, d))


# -- tree ensembles ------------------------------------------------------------------------------------------

_ENS_OPTS = ["Method", "NumLearningCycles", "Learners", "LearnRate", "Resample", "FResample", "Replace", "NPrint",
             "NumBins", "Options", "NumPredictorsToSample", "RatioToSmallest", "MarginPrecision", "RobustErrorGoal",
             "RobustMarginSigma", "RobustMaxMargin", "NPredToSample", "Type"]


def _tree_value(t: Tree) -> np.ndarray:
    return np.vstack(t.value)


class Ensemble(Model):
    def __init__(self, interp, d: Data, fname: str, classification: bool):
        self.classification = classification
        super().__init__(d, fname)
        self.interp = interp
        o = d.opts
        rng = rng_of(interp)
        K = d.labels.k if classification else 0
        methods = ["Bag", "AdaBoostM1", "AdaBoostM2", "LogitBoost", "GentleBoost", "LSBoost", "RUSBoost", "Subspace",
                   "TotalBoost", "LPBoost", "RobustBoost"]
        if "method" in o:
            method = choice(o["method"], methods, "Method")
        else:
            method = "lsboost" if not classification else ("logitboost" if K == 2 else "adaboostm2")
        if method in ("rusboost", "subspace", "totalboost", "lpboost", "robustboost"):
            raise MatlabError(f"{fname}: the method '{method}' is not supported.")
        if classification and method == "lsboost" or not classification and method not in ("bag", "lsboost"):
            raise MatlabError(f"{fname}: the method '{method}' does not apply to this kind of response.")
        if method in ("adaboostm1", "logitboost", "gentleboost") and K != 2:
            raise MatlabError(f"{fname}: the method '{method}' needs exactly two classes.")
        self.method = method
        self.matlab_class = ("ClassificationBaggedEnsemble" if method == "bag" else "ClassificationEnsemble") \
            if classification else ("RegressionBaggedEnsemble" if method == "bag" else "RegressionEnsemble")
        if not classification:
            self.default_loss = "mse"
        T = intval(o["numlearningcycles"], "NumLearningCycles") if "numlearningcycles" in o else 100
        lr = scalar(o["learnrate"], "LearnRate") if "learnrate" in o else 1.0
        kind, largs = learner_spec(o["learners"], fname) if "learners" in o else ("tree", [])
        if kind != "tree":
            raise MatlabError(f"{fname}: only tree learners are supported.")
        _, topts = get_opts(largs, 0, _TREE_OPTS + ["Type"], fname)
        bag = method == "bag"
        n, p = d.n, d.p
        defaults = {"minparent": 2 if (bag or classification) else 10, "minleaf": 1 if classification else 5,
                    "merge": False}
        if bag:
            defaults["nvars"] = max(1, int(np.ceil(np.sqrt(p)))) if classification else max(1, int(np.ceil(p / 3)))
            defaults["minparent"] = 2 if classification else 10
        else:
            defaults["maxsplits"] = 10
            defaults["minparent"] = 2
            defaults["minleaf"] = 1
        if "numpredictorstosample" in o or "npredtosample" in o:
            v = o.get("numpredictorstosample", o.get("npredtosample"))
            defaults["nvars"] = p if is_text(v) else min(intval(v, "NumPredictorsToSample"), p)
        self.tp = tree_params(topts, n, p, classification and method in ("bag", "adaboostm1", "adaboostm2"), fname,
                              defaults)
        resample = as_bool(o["resample"], "Resample") if "resample" in o else bag
        frac = scalar(o["fresample"], "FResample") if "fresample" in o else 1.0
        replace = as_bool(o["replace"], "Replace") if "replace" in o else True
        self.trees: list[Tree] = []
        self.alpha: list[float] = []
        self.oob: list[np.ndarray] = []
        X = d.X
        w = self.wn.copy()
        tp = self.tp

        def grow(target, weights, Kc, rows=None):
            rows = np.arange(n) if rows is None else rows
            return grow_tree(X[rows], target[rows], weights[rows], Kc, tp["crit"] if Kc else None, tp["minparent"],
                             tp["minleaf"], tp["maxsplits"], tp["nvars"], rng)

        def sample():
            m = max(1, int(round(frac * n)))
            return rng.choice(n, size=m, replace=replace, p=w / w.sum() if not bag else None) if resample else None

        if method == "bag":
            target = d.codes if classification else d.y
            for _ in range(T):
                rows = sample()
                self.trees.append(grow(target, self.wn, K or None, rows))
                mask = np.ones(n, dtype=bool)
                if rows is not None:
                    mask[rows] = False
                else:
                    mask[:] = False
                self.oob.append(mask)
                self.alpha.append(1.0)
        elif method == "adaboostm1":
            for _ in range(T):
                rows = sample()
                t = grow(d.codes, w if rows is None else np.ones(n), K, rows)
                pred = np.argmax(_tree_value(t)[t.apply(X)], axis=1)
                miss = pred != d.codes
                err = float(np.sum(w * miss) / w.sum())
                if err <= 0 or err >= 0.5:
                    if err <= 0:
                        self.trees.append(t)
                        self.alpha.append(1.0 if not self.trees[:-1] else max(self.alpha))
                    break
                a = 0.5 * lr * np.log((1 - err) / err)
                self.trees.append(t)
                self.alpha.append(a)
                w = w * np.exp(a * np.where(miss, 1.0, -1.0))
                w /= w.sum()
        elif method == "adaboostm2":
            Dm = np.tile((w / (K - 1))[:, None], (1, K))        # mislabel distribution over wrong labels
            Dm[np.arange(n), d.codes] = 0
            for _ in range(T):
                wi = Dm.sum(axis=1)
                rows = sample()
                t = grow(d.codes, wi / wi.sum() if rows is None else np.ones(n), K, rows)
                H = _tree_value(t)[t.apply(X)]
                ht = H[np.arange(n), d.codes]
                eps = 0.5 * float(np.sum(Dm * (1 - ht[:, None] + H)) / Dm.sum())
                if eps <= 0 or eps >= 0.5:
                    if eps <= 0:
                        self.trees.append(t)
                        self.alpha.append(1.0)
                    break
                beta = (eps / (1 - eps)) ** lr
                self.trees.append(t)
                self.alpha.append(float(np.log(1 / beta)))
                Dm = Dm * beta ** (0.5 * (1 + ht[:, None] - H))
                Dm[np.arange(n), d.codes] = 0
                Dm /= Dm.sum()
        elif method in ("logitboost", "gentleboost"):
            ypm = np.where(d.codes == 1, 1.0, -1.0)
            F = np.zeros(n)
            for _ in range(T):
                if method == "logitboost":
                    pr = 1 / (1 + np.exp(-2 * F))
                    wk = np.clip(pr * (1 - pr), 1e-10, None)
                    z = np.clip(((ypm + 1) / 2 - pr) / wk, -4, 4)
                    t = grow(z, self.wn * wk / np.sum(self.wn * wk), None)
                    f = 0.5 * np.array([v[0] for v in t.value])[t.apply(X)]
                else:
                    t = grow(ypm, w / w.sum(), None)
                    f = np.array([v[0] for v in t.value])[t.apply(X)]
                    w = w * np.exp(-ypm * f)
                    w /= w.sum()
                F += lr * f
                self.trees.append(t)
                self.alpha.append(lr * (0.5 if method == "logitboost" else 1.0))
        else:                                                       # least-squares boosting
            F = np.zeros(n)
            for _ in range(T):
                rows = sample()
                t = grow(d.y - F, self.wn, None, rows)
                F += lr * np.array([v[0] for v in t.value])[t.apply(X)]
                self.trees.append(t)
                self.alpha.append(lr)
        self.posterior_scores = classification and method == "bag"

    def refit(self, d: Data) -> "Ensemble":
        return Ensemble(self.interp, d, self.fname, self.classification)

    def _raw(self, X: np.ndarray, use: list | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        idx = range(len(self.trees)) if use is None else use
        K = self.d.labels.k if self.classification else 1
        out = np.zeros((X.shape[0], K))
        cnt = np.zeros(X.shape[0])
        for t_i in idx:
            t = self.trees[t_i]
            V = _tree_value(t)[t.apply(X)]
            sel = np.ones(X.shape[0], dtype=bool) if mask is None else mask[:, t_i]
            a = self.alpha[t_i]
            if self.method == "bag":
                out[sel] += V[sel]
                cnt[sel] += 1
            elif self.method == "adaboostm1":
                pred = np.argmax(V, axis=1)
                S = -np.ones((X.shape[0], K))
                S[np.arange(X.shape[0]), pred] = 1
                out += a * S
            elif self.method == "adaboostm2":
                out += a * V
            elif self.classification:
                out[:, 1] += a * V[:, 0]
                out[:, 0] -= a * V[:, 0]
            else:
                out[:, 0] += a * V[:, 0]
        if self.method == "bag":
            with np.errstate(all="ignore"):
                out = out / cnt[:, None]
        elif self.method == "adaboostm1" and self.alpha:
            out = out / max(sum(self.alpha), 1e-300)
        return out

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self._raw(X)

    def response(self, X: np.ndarray) -> np.ndarray:
        return self._raw(X)[:, 0]

    extra_props = ("NumTrained", "Trained", "TrainedWeights", "Method", "LearnerNames", "FitInfo")

    def get_field(self, interp, name: str) -> Any:
        if name == "NumTrained":
            return float(len(self.trees))
        if name == "TrainedWeights":
            return np.array(self.alpha).reshape(-1, 1)
        if name == "Method":
            return {"bag": "Bag", "adaboostm1": "AdaBoostM1", "adaboostm2": "AdaBoostM2", "logitboost": "LogitBoost",
                    "gentleboost": "GentleBoost", "lsboost": "LSBoost"}[self.method]
        if name == "LearnerNames":
            return cellstr(["Tree"], column=False)
        if name == "Trained":
            as_class = self.classification and self.method in ("bag", "adaboostm1", "adaboostm2")
            return Cell.from_list([TreeModel(self.interp, self.d, self.fname, as_class, self.tp, t)
                                   for t in self.trees], column=True)
        if name == "FitInfo":
            return np.zeros((len(self.trees), 0))
        return super().get_field(interp, name)

    def m_predict(self, interp, args, nargout):
        pos, opts = get_opts(args, 1, ["Learners", "UseObsForLearner"], "predict")
        if "useobsforlearner" in opts:
            raise MatlabError("predict: the option 'UseObsForLearner' is not supported.")
        X = self.d.encode_x(pos[0], "predict")
        use = [int(i) - 1 for i in vec(opts["learners"])] if "learners" in opts else None
        S = self._raw(X, use)
        if not self.classification:
            return S[:, :1]
        return outs(nargout, self.d.labels.decode(self.labels_from(S)), self.apply_transform(interp, S))

    def _cumulative(self, interp, X, y, w, lossfun) -> np.ndarray:
        out = []
        for k in range(1, len(self.trees) + 1):
            S = self._raw(X, list(range(k)))
            if self.classification:
                out.append(float(np.sum(w * (self.labels_from(S) != y)) / w.sum()))
            else:
                out.append(float(np.sum(w * (y - S[:, 0]) ** 2) / w.sum()))
        return np.array(out).reshape(-1, 1)

    def m_resubLoss(self, interp, args, nargout):
        _, opts = get_opts(args, 0, ["LossFun", "Mode", "Learners"], "resubLoss")
        y = self.d.codes if self.classification else self.d.y
        if "mode" in opts and choice(opts["mode"], ["ensemble", "cumulative", "individual"], "Mode") == "cumulative":
            return self._cumulative(interp, self.d.X, y, self.wn, None)
        return self.loss_value(interp, self.d.X, y, self.wn, opts.get("lossfun", char(self.default_loss)))

    def _oob_scores(self) -> np.ndarray:
        if self.method != "bag":
            raise MatlabError("Out-of-bag estimates are available for bagged ensembles only.")
        return self._raw(self.d.X, None, np.column_stack(self.oob))

    def m_oobPredict(self, interp, args, nargout):
        S = self._oob_scores()
        if not self.classification:
            return S[:, :1]
        S = np.where(np.isnan(S), self.prior[None, :], S)
        return outs(nargout, self.d.labels.decode(self.labels_from(S)), S)

    def m_oobLoss(self, interp, args, nargout):
        S = self._oob_scores()
        ok = ~np.isnan(S).any(axis=1)
        w = self.wn[ok] / self.wn[ok].sum()
        if self.classification:
            return float(np.sum(w * (self.labels_from(S[ok]) != self.d.codes[ok])))
        return float(np.sum(w * (self.d.y[ok] - S[ok, 0]) ** 2))

    def m_predictorImportance(self, interp, args, nargout):
        imp = np.zeros(self.d.p)
        for t in self.trees:
            nb = sum(1 for c in t.left if c >= 0)
            for i in range(t.n):
                if t.left[i] >= 0:
                    imp[t.var[i]] += (t.risk[i] - t.risk[t.left[i]] - t.risk[t.right[i]]) / max(nb, 1)
        return (imp / max(len(self.trees), 1)).reshape(1, -1)


@builtin("fitcensemble")
def _fitcensemble(interp, args, nargout):
    d = prepare(interp, args, "fitcensemble", _ENS_OPTS, True, drop_nan_x=False)
    return finish(interp, Ensemble(interp, d, "fitcensemble", True))


@builtin("fitrensemble")
def _fitrensemble(interp, args, nargout):
    d = prepare(interp, args, "fitrensemble", _ENS_OPTS, False, drop_nan_x=False)
    return finish(interp, Ensemble(interp, d, "fitrensemble", False))


@builtin("fitensemble")
def _fitensemble(interp, args, nargout):
    if len(args) < 5:
        raise MatlabError("fitensemble: expected fitensemble(X, Y, method, nlearn, learners, ...).")
    method = text(args[2])
    rest = [args[0], args[1], char("Method"), args[2], char("NumLearningCycles"), args[3], char("Learners"), args[4]]
    rest += list(args[5:])
    regression = method.lower() == "lsboost"
    for i in range(5, len(args) - 1):
        if is_text(args[i]) and text(args[i]).lower() == "type":
            regression = text(args[i + 1]).lower() == "regression"
    return (_fitrensemble if regression else _fitcensemble)(interp, rest, nargout)


class TreeBaggerModel(StatObject):
    matlab_class = "TreeBagger"

    def __init__(self, interp, ntrees: int, args: list):
        super().__init__()
        known = ["Method", "OOBPrediction", "OOBPred", "OOBPredictorImportance", "OOBVarImp", "MinLeafSize",
                 "MinLeaf", "NumPredictorsToSample", "NVarToSample", "InBagFraction", "FBoot",
                 "SampleWithReplacement", "SampleWithReplace", "MaxNumSplits", "SplitCriterion", "NumPrint",
                 "NPrint", "Options", "MergeLeaves", "Prune", "Surrogate", "PredictorSelection"]
        _, probe = get_opts(args, 2, COMMON_OPTS + known, "TreeBagger", text_pos=(2,))
        method = choice(probe["method"], ["classification", "regression"], "Method") if "method" in probe \
            else "classification"
        self.cls = method == "classification"
        d = prepare(interp, args, "TreeBagger", known, self.cls, drop_nan_x=False)
        if self.cls:      # class labels are always reported as text
            d.labels = _text_labels(d)
        o = d.opts
        self.d = d
        n, p = d.n, d.p
        ml = o.get("minleafsize", o.get("minleaf"))
        self.minleaf = intval(ml, "MinLeafSize") if ml is not None else (1 if self.cls else 5)
        nv = o.get("numpredictorstosample", o.get("nvartosample"))
        if nv is None:
            self.nvars = max(1, int(np.ceil(np.sqrt(p)))) if self.cls else max(1, int(np.ceil(p / 3)))
        else:
            self.nvars = p if is_text(nv) else min(intval(nv, "NumPredictorsToSample"), p)
        fb = o.get("inbagfraction", o.get("fboot"))
        self.frac = scalar(fb, "InBagFraction") if fb is not None else 1.0
        rp = o.get("samplewithreplacement", o.get("samplewithreplace"))
        self.replace = as_bool(rp, "SampleWithReplacement") if rp is not None else True
        ob = o.get("oobprediction", o.get("oobpred"))
        self.oob_on = as_bool(ob, "OOBPrediction") if ob is not None else False
        vi = o.get("oobpredictorimportance", o.get("oobvarimp"))
        if vi is not None and as_bool(vi, "OOBPredictorImportance"):
            self.oob_on = True
        crit = choice(o["splitcriterion"], ["gdi", "deviance", "twoing", "mse"], "SplitCriterion") \
            if "splitcriterion" in o else ("gdi" if self.cls else None)
        maxsplits = intval(o["maxnumsplits"], "MaxNumSplits") if "maxnumsplits" in o else n - 1
        rng = rng_of(interp)
        if self.cls:
            self.prior = class_prior(d, "TreeBagger")
            w = prior_weights(d, self.prior)
            target, K = d.codes, d.labels.k
        else:
            w = d.w / d.w.sum()
            target, K = d.y, None
        self.w = w
        self.trees, self.oob = [], []
        m = max(1, int(round(self.frac * n)))
        for _ in range(ntrees):
            rows = rng.choice(n, size=m, replace=self.replace)
            self.trees.append(grow_tree(d.X[rows], target[rows], w[rows], K, crit if self.cls else None,
                                        max(2 * self.minleaf, 2), self.minleaf, maxsplits, self.nvars, rng))
            mask = np.ones(n, dtype=bool)
            mask[rows] = False
            self.oob.append(mask)
        self.interp = interp

    def _scores(self, X: np.ndarray, mask: np.ndarray | None = None, upto: int | None = None):
        K = self.d.labels.k if self.cls else 1
        out = np.zeros((X.shape[0], K))
        sq = np.zeros(X.shape[0])
        cnt = np.zeros(X.shape[0])
        for k, t in enumerate(self.trees[:upto]):
            V = np.vstack(t.value)[t.apply(X)]
            sel = np.ones(X.shape[0], dtype=bool) if mask is None else mask[:, k]
            out[sel] += V[sel]
            sq[sel] += V[sel, 0] ** 2
            cnt[sel] += 1
        with np.errstate(all="ignore"):
            mean = out / cnt[:, None]
            sd = np.sqrt(np.clip(sq / cnt - mean[:, 0] ** 2, 0, None))
        return mean, sd

    def _labels(self, S: np.ndarray):
        S2 = np.where(np.isnan(S), self.prior[None, :], S)
        return self.d.labels.decode(np.argmax(S2, axis=1))

    def m_predict(self, interp, args, nargout):
        pos, opts = get_opts(args, 1, ["Trees", "TreeWeights", "UseInstanceForTree"], "predict")
        if opts:
            raise MatlabError("predict: tree selection options are not supported.")
        X = self.d.encode_x(pos[0], "predict")
        S, sd = self._scores(X)
        if not self.cls:
            return outs(nargout, S[:, :1], sd.reshape(-1, 1))
        return outs(nargout, self._labels(S), S, np.sqrt(S * (1 - S)))

    def _need_oob(self) -> np.ndarray:
        if not self.oob_on:
            raise MatlabError("Out-of-bag results need 'OOBPrediction', 'on' when the ensemble is grown.")
        return np.column_stack(self.oob)

    def m_oobPredict(self, interp, args, nargout):
        S, sd = self._scores(self.d.X, self._need_oob())
        if not self.cls:
            return outs(nargout, S[:, :1], sd.reshape(-1, 1))
        return outs(nargout, self._labels(S), S)

    def _error(self, X, y, mask, mode: str) -> Any:
        T = len(self.trees)
        ks = [T] if mode == "ensemble" else list(range(1, T + 1))
        out = []
        for k in ks:
            S, _ = self._scores(X, mask, k)
            ok = ~np.isnan(S).any(axis=1)
            if self.cls:
                out.append(float(np.mean(np.argmax(S[ok], axis=1) != y[ok])) if ok.any() else np.nan)
            else:
                out.append(float(np.mean((S[ok, 0] - y[ok]) ** 2)) if ok.any() else np.nan)
        return out[0] if mode == "ensemble" else np.array(out).reshape(-1, 1)

    def _mode(self, args: list, fname: str) -> str:
        _, opts = get_opts(args, 0, ["Mode", "Trees", "TreeWeights"], fname)
        if "trees" in opts or "treeweights" in opts:
            raise MatlabError(f"{fname}: tree selection options are not supported.")
        return choice(opts["mode"], ["cumulative", "ensemble", "individual"], "Mode") if "mode" in opts \
            else "cumulative"

    def m_oobError(self, interp, args, nargout):
        mode = self._mode(args, "oobError")
        if mode == "individual":
            raise MatlabError("oobError: 'Mode', 'individual' is not supported.")
        y = self.d.codes if self.cls else self.d.y
        return self._error(self.d.X, y, self._need_oob(), mode)

    def m_error(self, interp, args, nargout):
        if len(args) < 2:
            raise MatlabError("error: expected error(B, X, Y).")
        X = self.d.encode_x(args[0], "error")
        y = self.d.labels.encode(args[1]) if self.cls else vec(args[1])
        mode = self._mode(args[2:], "error")
        if mode == "individual":
            raise MatlabError("error: 'Mode', 'individual' is not supported.")
        return self._error(X, y, None, mode)

    def property_names(self) -> list[str]:
        return ["Trees", "NumTrees", "ClassNames", "Method", "OOBIndices", "InBagFraction", "MinLeafSize",
                "NumPredictorsToSample", "PredictorNames", "ResponseName", "X", "Y", "W", "ComputeOOBPrediction",
                "SampleWithReplacement", "NumPredictorSplit"]

    def get_field(self, interp, name: str) -> Any:
        d = self.d
        if name == "NumTrees":
            return float(len(self.trees))
        if name == "Trees":
            return Cell.from_list([TreeModel(self.interp, d, "TreeBagger", self.cls,
                                             {"minparent": 2, "minleaf": self.minleaf, "maxsplits": d.n,
                                              "nvars": self.nvars, "merge": False, "crit": "gdi"}, t)
                                   for t in self.trees], column=False)
        if name == "ClassNames":
            return cellstr(d.labels.names) if self.cls else cellstr([])
        if name == "Method":
            return "classification" if self.cls else "regression"
        if name == "OOBIndices":
            return np.column_stack(self.oob) if self.oob_on else np.zeros((0, 0))
        if name == "InBagFraction":
            return float(self.frac)
        if name == "MinLeafSize":
            return float(self.minleaf)
        if name == "NumPredictorsToSample":
            return float(self.nvars)
        if name == "PredictorNames":
            return cellstr(d.pnames, column=False)
        if name == "ResponseName":
            return d.rname
        if name == "X":
            return d.X
        if name == "Y":
            return d.labels.decode(d.codes) if self.cls else d.y.reshape(-1, 1)
        if name == "W":
            return self.w.reshape(-1, 1)
        if name == "ComputeOOBPrediction":
            return bool(self.oob_on)
        if name == "SampleWithReplacement":
            return bool(self.replace)
        if name == "NumPredictorSplit":
            cnt = np.zeros(d.p)
            for t in self.trees:
                for v in t.var:
                    if v >= 0:
                        cnt[v] += 1
            return cnt.reshape(1, -1)
        return super().get_field(interp, name)

    def display_lines(self) -> list[str]:
        return ["  TreeBagger", f"Ensemble with {len(self.trees)} bagged decision trees:",
                f"                    Training X:  [{self.d.n}x{self.d.p}]",
                f"                        Method:  {'classification' if self.cls else 'regression'}",
                f"                   MinLeafSize:  {self.minleaf}",
                f"         NumPredictorsToSample:  {self.nvars}"]


def _text_labels(d: Data) -> Labels:
    """Class labels as text (TreeBagger reports classes as a cell array of character vectors)."""
    old = d.labels
    names = old.names
    lab = Labels(cellstr([names[c] for c in d.codes]), cellstr(names))
    d.codes = lab.codes
    return lab


@builtin("TreeBagger")
def _treebagger(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("TreeBagger: expected TreeBagger(NumTrees, X, Y, 'Name', value, ...).")
    return TreeBaggerModel(interp, intval(args[0], "NumTrees"), list(args[1:]))


# -- Gaussian process regression -----------------------------------------------------------------------------

_GP_KERNELS = ["squaredexponential", "exponential", "matern32", "matern52", "rationalquadratic",
               "ardsquaredexponential", "ardexponential", "ardmatern32", "ardmatern52", "ardrationalquadratic"]


def gp_kernel(kind: str, A: np.ndarray, B: np.ndarray, theta: np.ndarray, d: int) -> np.ndarray:
    """Covariance between the rows of A and B; theta holds [length scale(s), (alpha), signal std]."""
    ard = kind.startswith("ard")
    base = kind[3:] if ard else kind
    nl = d if ard else 1
    ell = theta[:nl]
    sf = theta[-1]
    As, Bs = A / ell, B / ell
    r2 = np.clip(np.sum(As ** 2, axis=1)[:, None] + np.sum(Bs ** 2, axis=1)[None, :] - 2 * As @ Bs.T, 0, None)
    r = np.sqrt(r2)
    if base == "squaredexponential":
        k = np.exp(-0.5 * r2)
    elif base == "exponential":
        k = np.exp(-r)
    elif base == "matern32":
        k = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    elif base == "matern52":
        k = (1 + np.sqrt(5) * r + 5 * r2 / 3) * np.exp(-np.sqrt(5) * r)
    else:
        alpha = theta[nl]
        k = (1 + r2 / (2 * alpha)) ** (-alpha)
    return sf ** 2 * k


def _gp_basis(kind: str, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if kind == "none":
        return np.zeros((n, 0))
    if kind == "constant":
        return np.ones((n, 1))
    if kind == "linear":
        return np.column_stack([np.ones(n), X])
    return np.column_stack([np.ones(n), X, X ** 2])


class GPModel(Model):
    matlab_class = "RegressionGP"
    classification = False
    default_loss = "mse"
    extra_props = ("KernelFunction", "KernelInformation", "BasisFunction", "Beta", "Sigma", "LogLikelihood", "Alpha",
                   "ActiveSetVectors", "FitMethod", "PredictMethod", "PredictorLocation", "PredictorScale")

    def __init__(self, interp, d: Data, fname: str = "fitrgp"):
        from scipy.optimize import minimize
        super().__init__(d, fname)
        self.interp = interp
        o = d.opts
        self.kind = choice(o["kernelfunction"], _GP_KERNELS, "KernelFunction") if "kernelfunction" in o \
            else "squaredexponential"
        self.basis = choice(o["basisfunction"], ["none", "constant", "linear", "pureQuadratic"], "BasisFunction") \
            if "basisfunction" in o else "constant"
        self.basis = "purequadratic" if self.basis == "purequadratic" else self.basis
        fit = choice(o["fitmethod"], ["none", "exact", "sd", "sr", "fic"], "FitMethod") if "fitmethod" in o else "exact"
        if fit not in ("none", "exact"):
            raise MatlabError("fitrgp: only the 'exact' and 'none' fit methods are supported.")
        if "predictmethod" in o and choice(o["predictmethod"], ["exact", "bcd", "sd", "sr", "fic"],
                                           "PredictMethod") not in ("exact",):
            raise MatlabError("fitrgp: only the 'exact' prediction method is supported.")
        self.standardize = as_bool(o["standardize"], "Standardize") if "standardize" in o else False
        self.mu = d.X.mean(axis=0)
        sd = d.X.std(axis=0, ddof=1) if d.n > 1 else np.ones(d.p)
        self.sd = np.where(sd > 0, sd, 1.0)
        Z = (d.X - self.mu) / self.sd if self.standardize else d.X
        self.Z = Z
        y = d.y
        n, p = Z.shape
        ystd = float(np.std(y, ddof=1)) if n > 1 and np.std(y) > 0 else 1.0
        ard = self.kind.startswith("ard")
        zs = Z.std(axis=0, ddof=1) if n > 1 else np.ones(p)
        zs = np.where(zs > 0, zs, 1.0)
        ell0 = zs if ard else np.array([float(np.mean(zs))])
        theta0 = np.concatenate([ell0, [1.0] if "rationalquadratic" in self.kind else [], [ystd / np.sqrt(2)]])
        if "kernelparameters" in o:
            theta0 = vec(o["kernelparameters"])
            if theta0.size != (p if ard else 1) + (1 if "rationalquadratic" in self.kind
                    else 0) + 1 or np.any(theta0 <= 0):
                raise MatlabError("fitrgp: KernelParameters has the wrong number of (positive) values for this kernel.")
        sigma0 = scalar(o["sigma"], "Sigma") if "sigma" in o else ystd / np.sqrt(2)
        const_sigma = as_bool(o["constantsigma"], "ConstantSigma") if "constantsigma" in o else False
        slb = scalar(o["sigmalowerbound"], "SigmaLowerBound") if "sigmalowerbound" in o else 1e-2 * ystd
        const_k = np.zeros(theta0.size, dtype=bool)
        if "constantkernelparameters" in o:
            const_k = np.asarray(o["constantkernelparameters"]).astype(bool).flatten()
            if const_k.size != theta0.size:
                raise MatlabError("fitrgp: ConstantKernelParameters must have one flag per kernel parameter.")
        H = _gp_basis(self.basis, Z)
        self.H = H

        def unpack(v):
            th = theta0.copy()
            th[~const_k] = np.exp(v[:int(np.sum(~const_k))])
            sg = sigma0 if const_sigma else slb + np.exp(v[-1])
            return th, sg

        def solve(th, sg):
            K = gp_kernel(self.kind, Z, Z, th, p) + sg ** 2 * np.eye(n)
            L = np.linalg.cholesky(K + 1e-10 * np.trace(K) / n * np.eye(n))
            Li_y = np.linalg.solve(L, y)
            if H.shape[1]:
                Li_H = np.linalg.solve(L, H)
                beta, *_ = np.linalg.lstsq(Li_H, Li_y, rcond=None)
                res = Li_y - Li_H @ beta
            else:
                beta, res = np.zeros(0), Li_y
            ll = -0.5 * res @ res - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return ll, beta, L, res

        def nll(v):
            try:
                return -solve(*unpack(v))[0]
            except np.linalg.LinAlgError:
                return 1e100

        v0 = np.concatenate([np.log(theta0[~const_k]), [] if const_sigma else [np.log(max(sigma0 - slb, 1e-6 * ystd))]])
        if fit == "exact" and v0.size:
            best = minimize(nll, v0, method="L-BFGS-B", bounds=[(-12, 12)] * v0.size,
                            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
            v = best.x
        else:
            v = v0
        self.theta, self.sigma = unpack(v)
        self.ll, self.beta, self.L, res = solve(self.theta, self.sigma)
        self.alpha = np.linalg.solve(self.L.T, res)
        self.fit = fit

    def refit(self, d: Data) -> "GPModel":
        return GPModel(self.interp, d, self.fname)

    def _zmap(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd if self.standardize else X

    def response(self, X: np.ndarray) -> np.ndarray:
        Zq = self._zmap(X)
        Ks = gp_kernel(self.kind, Zq, self.Z, self.theta, self.d.p)
        return _gp_basis(self.basis, Zq) @ self.beta + Ks @ self.alpha

    def m_predict(self, interp, args, nargout):
        import scipy.stats as st
        pos, opts = get_opts(args, 1, ["Alpha"], "predict")
        X = self.d.encode_x(pos[0], "predict")
        y = self.response(X)
        if nargout < 2:
            return y.reshape(-1, 1)
        alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
        Zq = self._zmap(X)
        Ks = gp_kernel(self.kind, Zq, self.Z, self.theta, self.d.p)
        V = np.linalg.solve(self.L, Ks.T)
        var = self.theta[-1] ** 2 - np.sum(V ** 2, axis=0) + self.sigma ** 2
        sd = np.sqrt(np.clip(var, 0, None))
        z = st.norm.ppf(1 - alpha / 2)
        return outs(nargout, y.reshape(-1, 1), sd.reshape(-1, 1), np.column_stack([y - z * sd, y + z * sd]))

    def m_resubPredict(self, interp, args, nargout):
        return self.m_predict(interp, [self.d.X] + list(args), nargout)

    def get_field(self, interp, name: str) -> Any:
        if name == "KernelFunction":
            return {"squaredexponential": "SquaredExponential", "exponential": "Exponential", "matern32": "Matern32",
                    "matern52": "Matern52", "rationalquadratic": "RationalQuadratic",
                    "ardsquaredexponential": "ARDSquaredExponential", "ardexponential": "ARDExponential",
                    "ardmatern32": "ARDMatern32", "ardmatern52": "ARDMatern52",
                    "ardrationalquadratic": "ARDRationalQuadratic"}[self.kind]
        if name == "KernelInformation":
            ard = self.kind.startswith("ard")
            nl = self.d.p if ard else 1
            names = ([f"LengthScale{j + 1}" for j in range(nl)] if ard else ["LengthScale"]) + \
                (["AlphaRQ"] if "rationalquadratic" in self.kind else []) + ["SigmaF"]
            return struct(Name=self.get_field(interp, "KernelFunction"), KernelParameters=self.theta.reshape(-1, 1),
                          KernelParameterNames=cellstr(names))
        if name == "BasisFunction":
            return {"none": "None", "constant": "Constant", "linear": "Linear",
                    "purequadratic": "PureQuadratic"}[self.basis]
        if name == "Beta":
            return self.beta.reshape(-1, 1)
        if name == "Sigma":
            return float(self.sigma)
        if name == "LogLikelihood":
            return float(self.ll)
        if name == "Alpha":
            return self.alpha.reshape(-1, 1)
        if name == "ActiveSetVectors":
            return self.Z
        if name == "FitMethod":
            return "Exact" if self.fit == "exact" else "None"
        if name == "PredictMethod":
            return "Exact"
        if name == "PredictorLocation":
            return self.mu.reshape(1, -1) if self.standardize else np.zeros((0, 0))
        if name == "PredictorScale":
            return self.sd.reshape(1, -1) if self.standardize else np.zeros((0, 0))
        return super().get_field(interp, name)


@builtin("fitrgp")
def _fitrgp(interp, args, nargout):
    known = ["KernelFunction", "KernelParameters", "BasisFunction", "Beta", "Sigma", "ConstantSigma",
             "SigmaLowerBound", "ConstantKernelParameters", "Standardize", "FitMethod", "PredictMethod",
             "ActiveSet", "ActiveSetSize", "ActiveSetMethod", "Optimizer", "OptimizerOptions", "Verbose",
             "ComputationMethod", "DistanceMethod", "Regularization", "InitialStepSize", "CacheSize"]
    d = prepare(interp, args, "fitrgp", known, False)
    for bad in ("activeset", "beta"):
        if bad in d.opts:
            raise MatlabError(f"fitrgp: the option '{bad}' is not supported.")
    return finish(interp, GPModel(interp, d))


# -- small fully connected networks --------------------------------------------------------------------------

_NET_OPTS = ["LayerSizes", "Activations", "Lambda", "IterationLimit", "Standardize", "LayerWeightsInitializer",
             "LayerBiasesInitializer", "GradientTolerance", "LossTolerance", "StepTolerance", "Verbose",
             "VerboseFrequency", "ValidationData", "ValidationFrequency", "ValidationPatience", "StoreHistory"]
_ACT = {"relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(float)),
        "tanh": (np.tanh, lambda z, a: 1 - a ** 2),
        "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
        "none": (lambda z: z, lambda z, a: np.ones_like(z))}


class NetModel(Model):
    extra_props = ("LayerSizes", "LayerWeights", "LayerBiases", "Activations", "OutputLayerActivation",
                   "TrainingHistory", "ConvergenceInfo")

    def __init__(self, interp, d: Data, fname: str, classification: bool):
        from scipy.optimize import minimize
        self.classification = classification
        self.matlab_class = "ClassificationNeuralNetwork" if classification else "RegressionNeuralNetwork"
        if not classification:
            self.default_loss = "mse"
        super().__init__(d, fname)
        self.interp = interp
        o = d.opts
        if "validationdata" in o:
            raise MatlabError(f"{fname}: the option 'ValidationData' is not supported.")
        sizes = [int(v) for v in vec(o["layersizes"])] if "layersizes" in o else [10]
        acts = o.get("activations", char("relu"))
        from ._stats_util import text_list
        names = text_list(acts) if not is_text(acts) else [text(acts)] * len(sizes)
        if names is None or len(names) != len(sizes) or any(a.lower() not in _ACT for a in names):
            raise MatlabError(f"{fname}: Activations must be 'relu', 'tanh', 'sigmoid' or 'none' (one per layer).")
        self.acts = [a.lower() for a in names]
        lam = scalar(o["lambda"], "Lambda") if "lambda" in o else 0.0
        iters = intval(o["iterationlimit"], "IterationLimit") if "iterationlimit" in o else 1000
        self.standardize = as_bool(o["standardize"], "Standardize") if "standardize" in o else False
        self.mu = d.X.mean(axis=0)
        sd = d.X.std(axis=0, ddof=1) if d.n > 1 else np.ones(d.p)
        self.sd = np.where(sd > 0, sd, 1.0)
        Z = (d.X - self.mu) / self.sd if self.standardize else d.X
        K = d.labels.k if classification else 1
        dims = [d.p] + sizes + [K]
        self.sizes = sizes
        rng = rng_of(interp)
        shapes = [(dims[i + 1], dims[i]) for i in range(len(dims) - 1)]
        init = []
        for (r, c) in shapes:
            lim = np.sqrt(6.0 / (r + c))
            init.append(rng.uniform(-lim, lim, size=(r, c)).ravel())
            init.append(np.zeros(r))
        x0 = np.concatenate(init)
        w = self.wn
        if classification:
            Yoh = np.zeros((d.n, K))
            Yoh[np.arange(d.n), d.codes] = 1
        else:
            Yoh = d.y.reshape(-1, 1)

        def unpack(v):
            Ws, bs, k = [], [], 0
            for (r, c) in shapes:
                Ws.append(v[k:k + r * c].reshape(r, c))
                k += r * c
                bs.append(v[k:k + r])
                k += r
            return Ws, bs

        def forward(v, Xin):
            Ws, bs = unpack(v)
            A, cache = Xin, []
            for i, (W, b) in enumerate(zip(Ws, bs)):
                Zl = A @ W.T + b
                if i < len(Ws) - 1:
                    An = _ACT[self.acts[i]][0](Zl)
                else:
                    An = Zl
                cache.append((A, Zl, An))
                A = An
            return A, cache, Ws

        def loss(v):
            out, cache, Ws = forward(v, Z)
            if classification:
                m = out - out.max(axis=1, keepdims=True)
                P = np.exp(m)
                P /= P.sum(axis=1, keepdims=True)
                f = -np.sum(w * np.log(np.maximum(P[np.arange(d.n), d.codes], 1e-300)))
                delta = (P - Yoh) * w[:, None]
            else:
                e = out - Yoh
                f = 0.5 * np.sum(w * e[:, 0] ** 2)
                delta = e * w[:, None]
            grads = []
            for i in range(len(Ws) - 1, -1, -1):
                A, Zl, An = cache[i]
                gW = delta.T @ A + lam * Ws[i]
                grads.append((gW, delta.sum(axis=0)))
                f += 0.5 * lam * np.sum(Ws[i] ** 2)
                if i > 0:
                    Ap, Zp, Anp = cache[i - 1]
                    delta = (delta @ Ws[i]) * _ACT[self.acts[i - 1]][1](Zp, Anp)
            g = np.concatenate([np.concatenate([gW.ravel(), gb]) for gW, gb in reversed(grads)])
            return f, g

        sol = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": iters, "ftol": 1e-12, "gtol": 1e-8})
        self.v = sol.x
        self._forward = forward
        self._unpack = unpack
        self.iters = int(sol.nit)

    def refit(self, d: Data) -> "NetModel":
        return NetModel(self.interp, d, self.fname, self.classification)

    def _out(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd if self.standardize else X
        return self._forward(self.v, Z)[0]

    def scores(self, X: np.ndarray) -> np.ndarray:
        out = self._out(X)
        m = out - out.max(axis=1, keepdims=True)
        P = np.exp(m)
        return P / P.sum(axis=1, keepdims=True)

    def response(self, X: np.ndarray) -> np.ndarray:
        return self._out(X)[:, 0]

    def get_field(self, interp, name: str) -> Any:
        Ws, bs = self._unpack(self.v)
        if name == "LayerSizes":
            return np.array([self.sizes], dtype=float)
        if name == "LayerWeights":
            return Cell.from_list([W.copy() for W in Ws])
        if name == "LayerBiases":
            return Cell.from_list([b.reshape(-1, 1).copy() for b in bs])
        if name == "Activations":
            return self.acts[0] if len(self.acts) == 1 else cellstr(self.acts, column=False)
        if name == "OutputLayerActivation":
            return "softmax" if self.classification else "none"
        if name == "ConvergenceInfo":
            return struct(Iterations=float(self.iters))
        if name == "TrainingHistory":
            return np.zeros((0, 0))
        return super().get_field(interp, name)


@builtin("fitcnet")
def _fitcnet(interp, args, nargout):
    return finish(interp, NetModel(interp, prepare(interp, args, "fitcnet", _NET_OPTS, True), "fitcnet", True))


@builtin("fitrnet")
def _fitrnet(interp, args, nargout):
    return finish(interp, NetModel(interp, prepare(interp, args, "fitrnet", _NET_OPTS, False), "fitrnet", False))

