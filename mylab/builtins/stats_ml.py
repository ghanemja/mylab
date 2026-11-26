"""Classification and regression models: the shared model interface (predict, loss, cross-validation),
nearest neighbours, decision trees, naive Bayes, discriminant analysis and model assessment tools."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, Struct, char
from ._registry import builtin
from ._stats_util import (
    Labels, StatObject, as_bool, builtin_call, call, cellstr, choice, get_opts, given, intval, is_categorical,
    is_empty_arg, is_table, outs, rng_of, scalar, struct, text_list, vec,
)
from ._util import fnum, is_text, text

COMMON_OPTS = ["PredictorNames", "ResponseName", "ClassNames", "Prior", "Cost", "Weights", "ScoreTransform",
               "CategoricalPredictors", "CrossVal", "KFold", "Holdout", "Leaveout", "CVPartition",
               "OptimizeHyperparameters", "HyperparameterOptimizationOptions", "ResponseTransform"]
_CV_KEYS = ("crossval", "kfold", "holdout", "leaveout", "cvpartition")


class Data:
    """Training data of a model: numeric predictor matrix, response and observation weights."""

    def __init__(self):
        self.X = np.zeros((0, 0))
        self.pnames: list[str] = []
        self.levels: list[list[str] | None] = []   # category names of predictors that were not numeric
        self.rname = "Y"
        self.labels: Labels | None = None          # classification
        self.codes = np.zeros(0, dtype=int)
        self.y = np.zeros(0)                       # regression
        self.w = np.zeros(0)
        self.opts: dict = {}

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def subset(self, rows: np.ndarray) -> "Data":
        import copy
        d = copy.copy(self)
        d.X, d.w = self.X[rows], self.w[rows]
        if self.labels is not None:
            d.codes = self.codes[rows]
        else:
            d.y = self.y[rows]
        return d

    def encode_x(self, Xnew: Any, fname: str) -> np.ndarray:
        if is_table(Xnew):
            cols = dict(zip(Xnew.names, Xnew.columns))
            out = []
            for nme, lv in zip(self.pnames, self.levels):
                if nme not in cols:
                    raise MatlabError(f"{fname}: the table has no predictor variable '{nme}'.")
                out.append(_encode_col(cols[nme], lv))
            return np.column_stack(out) if out else np.zeros((Xnew.nrows, 0))
        X = np.asarray(fnum(Xnew), dtype=float)
        if X.ndim != 2 or X.shape[1] != self.p:
            if X.ndim == 2 and X.shape[1] == 1 and X.shape[0] == self.p and self.p > 1:
                return X.T
            raise MatlabError(f"{fname}: X must have {self.p} column(s), one per predictor.")
        return X


def _encode_col(c: Any, levels: list[str] | None) -> np.ndarray:
    if levels is None:
        return np.asarray(fnum(c), dtype=float).flatten(order="F")
    names = list(c.names_f()) if is_categorical(c) else (text_list(c) if text_list(c) is not None
                                                         else [str(int(v)) for v in vec(c)])
    return np.array([float(levels.index(s)) + 1 if s in levels else np.nan for s in names])


def prepare(interp, args: list, fname: str, extra: list, classification: bool, drop_nan_x: bool = True) -> Data:
    """Parse (X, Y, ...) or (Tbl, response, ...) and the common name-value options."""
    if len(args) < 2 and not (args and is_table(args[0])):
        raise MatlabError(f"{fname}: expected {fname}(X, Y, ...) or {fname}(Tbl, ResponseVarName, ...).")
    pos, opts = get_opts(args, 2 if len(args) >= 2 else 1, COMMON_OPTS + extra, fname)
    for o in ("optimizehyperparameters", "hyperparameteroptimizationoptions", "responsetransform"):
        if o in opts and not (is_text(opts[o]) and text(opts[o]).lower() == "none"):
            raise MatlabError(f"{fname}: the option '{o}' is not supported.")
    if len(pos) != 2:
        raise MatlabError(f"{fname}: expected {fname}(X, Y, 'Name', value, ...).")
    d = Data()
    d.opts = opts
    if is_table(pos[0]):
        tbl = pos[0]
        cols = dict(zip(tbl.names, tbl.columns))
        resp = pos[1]
        use = None
        if is_text(resp) and "~" in text(resp):
            lhs, rhs = text(resp).split("~", 1)
            d.rname = lhs.strip()
            use = [t.strip() for t in rhs.split("+") if t.strip() and t.strip() != "1"]
            for u in use:
                if u not in cols:
                    raise MatlabError(f"{fname}: the formula uses '{u}', which is not a table variable.")
            yv = cols.get(d.rname)
        elif is_text(resp) and text(resp) in cols:
            d.rname = text(resp)
            yv = cols[d.rname]
        else:
            yv = resp
            d.rname = "Y"
        if yv is None:
            raise MatlabError(f"{fname}: '{d.rname}' is not a table variable.")
        names = use if use is not None else [n for n in tbl.names if n != d.rname]
        if "predictornames" in opts:
            names = text_list(opts["predictornames"]) or names
        xs = []
        for nme in names:
            c = cols[nme]
            if isinstance(c, np.ndarray) and c.dtype.kind in "iuf":
                d.levels.append(None)
                xs.append(np.asarray(c, dtype=float).reshape(tbl.nrows, -1)[:, 0])
            else:
                lab = Labels(c)
                d.levels.append(list(lab.names))
                xs.append(np.where(lab.codes >= 0, lab.codes + 1.0, np.nan))
        d.X = np.column_stack(xs) if xs else np.zeros((tbl.nrows, 0))
        d.pnames = list(names)
    else:
        X = np.asarray(fnum(pos[0]), dtype=float)
        if X.ndim != 2:
            raise MatlabError(f"{fname}: X must be a matrix with one observation per row.")
        d.X = X
        d.pnames = [f"x{j + 1}" for j in range(X.shape[1])]
        if "predictornames" in opts:
            d.pnames = text_list(opts["predictornames"]) or []
            if len(d.pnames) != X.shape[1]:
                raise MatlabError(f"{fname}: PredictorNames must list one name per column of X.")
        d.levels = [None] * X.shape[1]
        yv = pos[1]
    if "responsename" in opts:
        d.rname = text(opts["responsename"])
    n = d.X.shape[0]
    w = vec(opts["weights"]) if "weights" in opts else np.ones(n)
    if w.size != n or np.any(w < 0):
        raise MatlabError(f"{fname}: Weights must hold one non-negative value per observation.")
    keep = ~np.isnan(w)
    if drop_nan_x:
        keep &= ~np.isnan(d.X).any(axis=1)
    if classification:
        lab = Labels(yv, opts.get("classnames"))
        if lab.n != n:
            raise MatlabError(f"{fname}: Y must have one label per row of X.")
        keep &= lab.codes >= 0
        d.labels = lab
        d.codes = lab.codes[keep]
        if lab.k < 1:
            raise MatlabError(f"{fname}: Y has no valid class labels.")
    else:
        y = vec(yv)
        if y.size != n:
            raise MatlabError(f"{fname}: Y must have one value per row of X.")
        keep &= ~np.isnan(y)
        d.y = y[keep]
    d.X, d.w = d.X[keep], w[keep]
    if d.X.shape[0] == 0:
        raise MatlabError(f"{fname}: there are no observations without missing values.")
    return d


def class_prior(d: Data, fname: str) -> np.ndarray:
    k = d.labels.k
    counts = np.array([d.w[d.codes == c].sum() for c in range(k)])
    v = d.opts.get("prior")
    if v is None or (is_text(v) and text(v).lower() == "empirical"):
        p = counts.copy()
    elif is_text(v):
        if text(v).lower() != "uniform":
            raise MatlabError(f"{fname}: Prior must be 'empirical', 'uniform', a vector or a structure.")
        p = np.ones(k)
    elif isinstance(v, Struct):
        p = vec(v.get_field(0, "ClassProbs"))
    else:
        p = vec(v)
    if p.size != k or np.any(p < 0) or p.sum() <= 0:
        raise MatlabError(f"{fname}: Prior must hold one non-negative value per class.")
    return p / p.sum()


def class_cost(d: Data, fname: str) -> np.ndarray:
    k = d.labels.k
    v = d.opts.get("cost")
    if v is None:
        return 1.0 - np.eye(k)
    C = np.asarray(fnum(v.get_field(0, "ClassificationCosts") if isinstance(v, Struct) else v), dtype=float)
    if C.shape != (k, k):
        raise MatlabError(f"{fname}: Cost must be a {k}-by-{k} matrix.")
    return C


def prior_weights(d: Data, prior: np.ndarray) -> np.ndarray:
    """Observation weights rescaled so that the weights of each class add up to its prior probability."""
    w = d.w.astype(float).copy()
    for c in range(prior.size):
        sel = d.codes == c
        tot = w[sel].sum()
        if tot > 0:
            w[sel] *= prior[c] / tot
    return w / w.sum()


_TRANSFORMS: dict[str, Callable] = {
    "none": lambda s: s, "identity": lambda s: s,
    "doublelogit": lambda s: 1 / (1 + np.exp(-2 * s)), "logit": lambda s: 1 / (1 + np.exp(-s)),
    "invlogit": lambda s: np.log(s / (1 - s)), "sign": np.sign, "symmetric": lambda s: 2 * s - 1,
    "symmetriclogit": lambda s: 2 / (1 + np.exp(-s)) - 1,
    "ismax": lambda s: (s == s.max(axis=1, keepdims=True)).astype(float),
    "symmetricismax": lambda s: 2 * (s == s.max(axis=1, keepdims=True)).astype(float) - 1,
}


# -- model base class -----------------------------------------------------------------------------------------

class Model(StatObject):
    """Common behaviour of the fitted classification and regression models."""

    classification = True
    posterior_scores = False      # scores are class probabilities (labels then minimise the expected cost)
    default_loss = "classiferror"

    def __init__(self, d: Data, fname: str):
        super().__init__()
        self.d = d
        self.fname = fname
        self.transform: Any = "none"
        if self.classification:
            self.prior = class_prior(d, fname)
            self.cost = class_cost(d, fname)
            self.wn = prior_weights(d, self.prior)
            if "scoretransform" in d.opts:
                self.set_transform(d.opts["scoretransform"])
        else:
            self.wn = d.w / d.w.sum()

    # -- to be provided by each learner ---------------------------------------------------------------
    def scores(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def response(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def refit(self, d: Data) -> "Model":
        raise NotImplementedError

    # -- shared helpers -----------------------------------------------------------------------------------
    def set_transform(self, v: Any) -> None:
        if isinstance(v, FunctionHandle):
            self.transform = v
        else:
            name = text(v).lower()
            if name not in _TRANSFORMS:
                raise MatlabError(f"Unknown ScoreTransform '{text(v)}'.")
            self.transform = name

    def apply_transform(self, interp, S: np.ndarray) -> np.ndarray:
        if isinstance(self.transform, FunctionHandle):
            return np.asarray(fnum(call(interp, self.transform, [S], 1)[0]), dtype=float)
        with np.errstate(all="ignore"):
            return _TRANSFORMS[self.transform](S)

    def labels_from(self, S: np.ndarray) -> np.ndarray:
        if self.posterior_scores:
            return np.argmin(S @ self.cost, axis=1)
        return np.argmax(S, axis=1)

    def classify(self, interp, X: np.ndarray):
        S = self.scores(X)
        codes = self.labels_from(S)
        codes = np.where(np.isnan(S).all(axis=1), int(np.argmax(self.prior)), codes)
        return codes, self.apply_transform(interp, S), S

    def _xy(self, args: list, fname: str):
        """(X, Y) for loss-like methods: (X, Y) or (Tbl, response name) or (Tbl, Y)."""
        if len(args) < 2 and not (args and is_table(args[0])):
            raise MatlabError(f"{fname}: expected {fname}(mdl, X, Y).")
        X = self.d.encode_x(args[0], fname)
        if len(args) < 2 or (is_table(args[0]) and is_text(args[1]) and text(args[1]) in args[0].names
                             and X.shape[0] != 1):
            cols = dict(zip(args[0].names, args[0].columns))
            name = text(args[1]) if len(args) > 1 else self.d.rname
            if name not in cols:
                raise MatlabError(f"{fname}: the table has no response variable '{name}'.")
            yv = cols[name]
            rest = list(args[2:])
        else:
            yv = args[1]
            rest = list(args[2:])
        if self.classification:
            y = self.d.labels.encode(yv)
        else:
            y = vec(yv)
        if y.size != X.shape[0]:
            raise MatlabError(f"{fname}: Y must have one value per row of X.")
        return X, y, rest

    def loss_value(self, interp, X, y, w, lossfun: Any) -> float:
        ok = (y >= 0) if self.classification else ~np.isnan(y)
        X, y, w = X[ok], y[ok], w[ok]
        if w.sum() <= 0:
            return np.nan
        w = w / w.sum()
        if not self.classification:
            f = self.response(X)
            if isinstance(lossfun, FunctionHandle):
                return float(fnum(call(interp, lossfun, [y.reshape(-1, 1), f.reshape(-1, 1), w.reshape(-1, 1)],
                        1)[0]).flat[0])
            name = choice(lossfun, ["mse", "epsiloninsensitive"], "LossFun")
            if name == "mse":
                return float(np.sum(w * (y - f) ** 2))
            eps = getattr(self, "epsilon", 0.0)
            return float(np.sum(w * np.maximum(np.abs(y - f) - eps, 0)))
        codes, St, S = self.classify(interp, X)
        k = self.d.labels.k
        if isinstance(lossfun, FunctionHandle):
            Cm = np.zeros((y.size, k))
            Cm[np.arange(y.size), y] = 1
            return float(fnum(call(interp, lossfun, [Cm.astype(bool), St, w.reshape(-1, 1), self.cost], 1)[0]).flat[0])
        name = choice(lossfun, ["classiferror", "mincost", "classifcost", "binodeviance", "exponential", "hinge",
                                "logit", "quadratic", "crossentropy"], "LossFun")
        if name == "classiferror":
            return float(np.sum(w * (codes != y)))
        if name in ("mincost", "classifcost"):
            return float(np.sum(w * self.cost[y, codes]))
        m = St[np.arange(y.size), y]
        with np.errstate(all="ignore"):
            if name == "crossentropy":
                return float(-np.sum(w * np.log(np.maximum(m, 1e-300))))
            if name == "binodeviance":
                return float(np.sum(w * np.log1p(np.exp(-2 * m))))
            if name == "exponential":
                return float(np.sum(w * np.exp(-m)))
            if name == "hinge":
                return float(np.sum(w * np.maximum(0, 1 - m)))
            if name == "logit":
                return float(np.sum(w * np.log1p(np.exp(-m))))
            return float(np.sum(w * (1 - m) ** 2))

    # -- MATLAB methods ------------------------------------------------------------------------------------
    def m_predict(self, interp, args, nargout):
        if len(args) != 1:
            raise MatlabError("predict: expected predict(mdl, X).")
        X = self.d.encode_x(args[0], "predict")
        if not self.classification:
            return self.response(X).reshape(-1, 1)
        codes, St, S = self.classify(interp, X)
        res = [self.d.labels.decode(codes), St]
        if self.posterior_scores:
            res.append(S @ self.cost)
        return outs(nargout, *res)

    def m_resubPredict(self, interp, args, nargout):
        if args:
            raise MatlabError("resubPredict: options are not supported.")
        return self.m_predict(interp, [self.d.X], nargout)

    def _loss_opts(self, rest: list, n: int, fname: str):
        _, opts = get_opts(rest, 0, ["LossFun", "Weights", "Mode", "Learners", "UseObsForLearner", "Subtrees",
                                     "IncludeInteractions"], fname)
        for o in ("mode", "learners", "useobsforlearner", "subtrees", "includeinteractions"):
            if o in opts:
                raise MatlabError(f"{fname}: the option '{o}' is not supported for this model.")
        w = vec(opts["weights"]) if "weights" in opts else None
        if w is not None and w.size != n:
            raise MatlabError(f"{fname}: Weights must hold one value per observation.")
        return opts.get("lossfun", char(self.default_loss)), w

    def m_loss(self, interp, args, nargout):
        X, y, rest = self._xy(args, "loss")
        lossfun, w = self._loss_opts(rest, y.size, "loss")
        if w is None:
            w = np.ones(y.size)
            if self.classification:   # weight the classes by their prior probabilities
                for c in range(self.d.labels.k):
                    sel = y == c
                    if sel.any():
                        w[sel] = self.prior[c] / sel.sum()
        return self.loss_value(interp, X, y, w, lossfun)

    def m_resubLoss(self, interp, args, nargout):
        lossfun, w = self._loss_opts(list(args), self.d.n, "resubLoss")
        y = self.d.codes if self.classification else self.d.y
        return self.loss_value(interp, self.d.X, y, self.wn if w is None else w, lossfun)

    def _margins(self, interp, X, y):
        if not self.classification:
            raise MatlabError("margin is defined for classification models only.")
        _, St, _ = self.classify(interp, X)
        true = St[np.arange(y.size), y]
        other = St.copy()
        other[np.arange(y.size), y] = -np.inf
        return true - other.max(axis=1)

    def m_margin(self, interp, args, nargout):
        X, y, _ = self._xy(args, "margin")
        return self._margins(interp, X, np.where(y < 0, 0, y)).reshape(-1, 1)

    def m_resubMargin(self, interp, args, nargout):
        return self._margins(interp, self.d.X, self.d.codes).reshape(-1, 1)

    def m_edge(self, interp, args, nargout):
        X, y, _ = self._xy(args, "edge")
        return float(np.mean(self._margins(interp, X, np.where(y < 0, 0, y))))

    def m_resubEdge(self, interp, args, nargout):
        return float(np.sum(self.wn * self._margins(interp, self.d.X, self.d.codes)))

    def m_crossval(self, interp, args, nargout):
        _, opts = get_opts(args, 0, ["KFold", "Holdout", "Leaveout", "CVPartition"], "crossval")
        return cross_validate(interp, self, opts)

    def m_compact(self, interp, args, nargout):
        return self

    # -- properties ---------------------------------------------------------------------------------------
    extra_props: tuple = ()

    def property_names(self) -> list[str]:
        base = ["X", "Y", "W", "NumObservations", "PredictorNames", "ResponseName", "ExpandedPredictorNames"]
        if self.classification:
            base += ["ClassNames", "Prior", "Cost", "ScoreTransform"]
        return base + list(self.extra_props)

    def get_field(self, interp, name: str) -> Any:
        d = self.d
        if name == "X":
            return d.X
        if name == "Y":
            return d.labels.decode(d.codes) if self.classification else d.y.reshape(-1, 1)
        if name == "W":
            return self.wn.reshape(-1, 1)
        if name == "NumObservations":
            return float(d.n)
        if name in ("PredictorNames", "ExpandedPredictorNames"):
            return cellstr(d.pnames, column=False)
        if name == "ResponseName":
            return d.rname
        if name == "CategoricalPredictors":
            idx = [j + 1.0 for j, lv in enumerate(d.levels) if lv is not None]
            return np.array([idx]) if idx else np.zeros((0, 0))
        if self.classification:
            if name == "ClassNames":
                return d.labels.class_value()
            if name == "Prior":
                return self.prior.reshape(1, -1)
            if name == "Cost":
                return self.cost
            if name == "ScoreTransform":
                return self.transform
        return super().get_field(interp, name)

    def set_field(self, interp, name: str, value: Any):
        if name == "ScoreTransform" and self.classification:
            m = self.clone()
            m.set_transform(value)
            return m
        return super().set_field(interp, name, value)

    def display_lines(self) -> list[str]:
        d = self.d
        lines = [f"  {self.matlab_class}", f"             ResponseName: '{d.rname}'"]
        if self.classification:
            lines.append("               ClassNames: " + " ".join(d.labels.names))
            st = repr(self.transform) if isinstance(self.transform, FunctionHandle) else f"'{self.transform}'"
            lines.append(f"           ScoreTransform: {st}")
        lines.append(f"          NumObservations: {d.n}")
        return lines


# -- cross-validated models -------------------------------------------------------------------------------------

class PartitionedModel(StatObject):
    def __init__(self, model: Model, part, trained: list):
        super().__init__()
        self.model, self.part, self.trained = model, part, trained
        kind = "Classification" if model.classification else "Regression"
        self.matlab_class = f"{kind}PartitionedModel"

    def property_names(self) -> list[str]:
        return ["Trained", "KFold", "Partition", "NumObservations", "X", "Y", "W", "PredictorNames", "ResponseName",
                "CrossValidatedModel"] + (["ClassNames", "Prior", "Cost"] if self.model.classification else [])

    def get_field(self, interp, name: str) -> Any:
        if name == "Trained":
            return Cell.from_list(self.trained, column=True)
        if name == "KFold":
            return float(len(self.trained))
        if name == "Partition":
            return self.part
        if name == "CrossValidatedModel":
            return self.model.matlab_class.replace("Classification", "").replace("Regression", "")
        return self.model.get_field(interp, name)

    def _fold_predictions(self, interp):
        m = self.model
        n = m.d.n
        k = m.d.labels.k if m.classification else 1
        S = np.full((n, k), np.nan)
        codes = np.full(n, -1)
        for mdl, (_, te) in zip(self.trained, self.part.train_test()):
            if m.classification:
                c, St, _ = mdl.classify(interp, m.d.X[te])
                S[te], codes[te] = St, c
            else:
                S[te, 0] = mdl.response(m.d.X[te])
        return codes, S

    def m_kfoldPredict(self, interp, args, nargout):
        if args:
            raise MatlabError("kfoldPredict: options are not supported.")
        codes, S = self._fold_predictions(interp)
        if not self.model.classification:
            return S
        lab = self.model.d.labels
        safe = np.where(codes < 0, int(np.argmax(self.model.prior)), codes)
        return outs(nargout, lab.decode(safe), S)

    def m_kfoldLoss(self, interp, args, nargout):
        _, opts = get_opts(args, 0, ["LossFun", "Mode", "Folds"], "kfoldLoss")
        mode = choice(opts["mode"], ["average", "individual"], "Mode") if "mode" in opts else "average"
        m = self.model
        lossfun = opts.get("lossfun", char(m.default_loss))
        y = m.d.codes if m.classification else m.d.y
        pairs = list(zip(self.trained, self.part.train_test()))
        if "folds" in opts:
            pairs = [pairs[int(i) - 1] for i in vec(opts["folds"])]
        vals, sizes = [], []
        for mdl, (_, te) in pairs:
            w = m.wn[te]
            vals.append(mdl.loss_value(interp, m.d.X[te], y[te], w, lossfun))
            sizes.append(w.sum())
        vals, sizes = np.array(vals), np.array(sizes)
        if mode == "individual":
            return vals.reshape(-1, 1)
        ok = ~np.isnan(vals)
        return float(np.sum(vals[ok] * sizes[ok]) / np.sum(sizes[ok])) if ok.any() else np.nan

    def m_kfoldMargin(self, interp, args, nargout):
        m = self.model
        out = np.full(m.d.n, np.nan)
        for mdl, (_, te) in zip(self.trained, self.part.train_test()):
            out[te] = mdl._margins(interp, m.d.X[te], m.d.codes[te])
        return out.reshape(-1, 1)

    def m_kfoldEdge(self, interp, args, nargout):
        return float(np.nanmean(self.m_kfoldMargin(interp, [], 1)))

    def display_lines(self) -> list[str]:
        return [f"  {self.matlab_class}", f"    CrossValidatedModel: '{self.get_field(None, 'CrossValidatedModel')}'",
                f"        NumObservations: {self.model.d.n}", f"                  KFold: {len(self.trained)}"]


def cross_validate(interp, model: Model, opts: dict) -> PartitionedModel:
    from .stats_desc import make_partition, partition_from_opts
    n = model.d.n
    codes = model.d.codes if model.classification else None
    o = dict(opts)
    if "crossval" in o:
        on = as_bool(o.pop("crossval"), "CrossVal")
        if not on and not any(k in o for k in _CV_KEYS):
            raise MatlabError("crossval: 'CrossVal', 'off' does not make a cross-validated model.")
    part = partition_from_opts(interp, o, n, codes, model.fname)
    if part is None:
        part = make_partition(rng_of(interp), n, "kfold", float(min(10, n)), codes)
    trained = [model.refit(model.d.subset(tr)) for tr, _ in part.train_test()]
    return PartitionedModel(model, part, trained)


def finish(interp, model: Model):
    """Return the model, or its cross-validated version when a cross-validation option was given."""
    opts = model.d.opts
    if any(k in opts for k in _CV_KEYS):
        if "crossval" in opts and not as_bool(opts["crossval"], "CrossVal") and \
                not any(k in opts for k in _CV_KEYS[1:]):
            return model
        return cross_validate(interp, model, {k: opts[k] for k in _CV_KEYS if k in opts})
    return model


# -- k-nearest neighbours --------------------------------------------------------------------------------------------

_KNN_OPTS = ["NumNeighbors", "Distance", "DistanceWeight", "Standardize", "BreakTies", "Exponent", "Cov", "Scale",
             "IncludeTies", "NSMethod", "BucketSize"]


class KNNModel(Model):
    matlab_class = "ClassificationKNN"
    posterior_scores = True
    default_loss = "mincost"
    extra_props = ("NumNeighbors", "Distance", "DistanceWeight", "Mu", "Sigma", "BreakTies", "IncludeTies")

    def __init__(self, interp, d: Data, fname: str = "fitcknn"):
        super().__init__(d, fname)
        o = d.opts
        self.interp = interp
        self.k = intval(o["numneighbors"], "NumNeighbors") if "numneighbors" in o else 1
        if self.k < 1:
            raise MatlabError("fitcknn: NumNeighbors must be a positive integer.")
        from .stats_cluster import metric_name
        self.metric = metric_name(o["distance"], fname) if "distance" in o else "euclidean"
        dw = o.get("distanceweight", char("equal"))
        self.dweight = dw if isinstance(dw, FunctionHandle) else choice(dw, ["equal", "inverse", "squaredinverse"],
                                                                         "DistanceWeight")
        self.standardize = as_bool(o["standardize"], "Standardize") if "standardize" in o else False
        self.ties = choice(o["breakties"], ["smallest", "nearest", "random"], "BreakTies") if "breakties" in o \
            else "smallest"
        self.include_ties = as_bool(o["includeties"], "IncludeTies") if "includeties" in o else False
        if "nsmethod" in o:
            choice(o["nsmethod"], ["kdtree", "exhaustive"], "NSMethod")
        self.extra = []
        for key, met in (("exponent", "minkowski"), ("cov", "mahalanobis"), ("scale", "seuclidean")):
            if key in o:
                if self.metric != met:
                    raise MatlabError(f"fitcknn: '{key}' applies to the {met} distance only.")
                self.extra = [o[key]]
        self.mu = d.X.mean(axis=0)
        sd = d.X.std(axis=0, ddof=1) if d.n > 1 else np.ones(d.p)
        self.sigma = np.where(sd > 0, sd, 1.0)
        self.Z = (d.X - self.mu) / self.sigma if self.standardize else d.X

    def refit(self, d: Data) -> "KNNModel":
        return KNNModel(self.interp, d, self.fname)

    def scores(self, X: np.ndarray) -> np.ndarray:
        from .stats_cluster import dist_matrix
        Xq = (X - self.mu) / self.sigma if self.standardize else X
        ok = ~np.isnan(Xq).any(axis=1)
        K = self.d.labels.k
        S = np.full((X.shape[0], K), np.nan)
        if not ok.any():
            return S
        D = dist_matrix(self.interp, Xq[ok], self.Z, self.metric, self.extra, "predict", ref=self.Z)
        kk = min(self.k, self.d.n)
        order = np.argsort(D, axis=1, kind="stable")
        rows = np.flatnonzero(ok)
        for r, i in enumerate(rows):
            idx = order[r, :kk]
            if self.include_ties:
                idx = order[r, :int(np.sum(D[r] <= D[r, order[r, kk - 1]]))]
            dd = D[r, idx]
            with np.errstate(all="ignore"):
                if isinstance(self.dweight, FunctionHandle):
                    dwt = vec(call(self.interp, self.dweight, [dd.reshape(1, -1)], 1)[0])
                elif self.dweight == "inverse":
                    dwt = 1 / dd
                elif self.dweight == "squaredinverse":
                    dwt = 1 / dd ** 2
                else:
                    dwt = np.ones(dd.size)
            if np.any(np.isinf(dwt)):
                dwt = np.isinf(dwt).astype(float)
            wts = dwt * self.wn[idx]
            s = np.bincount(self.d.codes[idx], weights=wts, minlength=K)
            tot = s.sum()
            S[i] = s / tot if tot > 0 else self.prior
            if self.ties == "nearest" and np.sum(S[i] == S[i].max()) > 1:
                best = np.flatnonzero(S[i] == S[i].max())
                first = next(int(self.d.codes[j]) for j in idx if self.d.codes[j] in best)
                S[i, first] += 1e-12
        return S

    def get_field(self, interp, name: str) -> Any:
        if name == "NumNeighbors":
            return float(self.k)
        if name == "Distance":
            return self.metric
        if name == "DistanceWeight":
            return self.dweight
        if name == "Mu":
            return self.mu.reshape(1, -1) if self.standardize else np.zeros((0, 0))
        if name == "Sigma":
            return self.sigma.reshape(1, -1) if self.standardize else np.zeros((0, 0))
        if name == "BreakTies":
            return self.ties
        if name == "IncludeTies":
            return bool(self.include_ties)
        return super().get_field(interp, name)


@builtin("fitcknn")
def _fitcknn(interp, args, nargout):
    return finish(interp, KNNModel(interp, prepare(interp, args, "fitcknn", _KNN_OPTS, True)))


# -- decision trees ----------------------------------------------------------------------------------------------------

class Tree:
    """A binary decision tree stored as arrays (node 0 is the root)."""

    def __init__(self):
        self.left: list[int] = []
        self.right: list[int] = []
        self.var: list[int] = []
        self.cut: list[float] = []
        self.parent: list[int] = []
        self.value: list[np.ndarray] = []    # class probabilities, or [mean] for regression
        self.size: list[int] = []
        self.prob: list[float] = []
        self.risk: list[float] = []          # node error times node probability
        self.counts: list[np.ndarray] = []

    def add(self, parent: int, value, size, prob, risk, counts) -> int:
        self.left.append(-1)
        self.right.append(-1)
        self.var.append(-1)
        self.cut.append(np.nan)
        self.parent.append(parent)
        self.value.append(value)
        self.size.append(size)
        self.prob.append(prob)
        self.risk.append(risk)
        self.counts.append(counts)
        return len(self.left) - 1

    @property
    def n(self) -> int:
        return len(self.left)

    def apply(self, X: np.ndarray) -> np.ndarray:
        node = np.zeros(X.shape[0], dtype=int)
        left, right, var, cut = map(np.array, (self.left, self.right, self.var, self.cut))
        active = np.arange(X.shape[0])
        while active.size:
            nd = node[active]
            branch = left[nd] >= 0
            active, nd = active[branch], nd[branch]
            if not active.size:
                break
            x = X[active, var[nd]]
            ok = ~np.isnan(x)                     # a missing value stops at the current node
            go_left = x < cut[nd]
            node[active[ok]] = np.where(go_left[ok], left[nd[ok]], right[nd[ok]])
            active = active[ok]
        return node

    def compact(self, keep_leaf: np.ndarray) -> "Tree":
        """A copy in which the nodes flagged in ``keep_leaf`` become leaves (their subtrees are dropped)."""
        t = Tree()
        queue = [(0, -1)]
        new_id: dict = {}
        while queue:
            old, par = queue.pop(0)
            i = t.add(par, self.value[old], self.size[old], self.prob[old], self.risk[old], self.counts[old])
            new_id[old] = i
            if par >= 0:
                if t.left[par] < 0:
                    t.left[par] = i
                else:
                    t.right[par] = i
            if self.left[old] >= 0 and not keep_leaf[old]:
                t.var[i], t.cut[i] = self.var[old], self.cut[old]
                queue.append((self.left[old], i))
                queue.append((self.right[old], i))
        return t


def _impurity(P: np.ndarray, crit: str) -> np.ndarray:
    with np.errstate(all="ignore"):
        if crit == "deviance":
            return -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=-1)
        return 1 - np.sum(P ** 2, axis=-1)


def _best_split(X, rows, w, target, K, crit, minleaf, feats):
    """Best (feature, cut, gain) for the observations in ``rows``; None when no split is admissible."""
    best = None
    wr = w[rows]
    tot_w = wr.sum()
    if K is not None:
        T = np.zeros((rows.size, K))
        T[np.arange(rows.size), target[rows]] = wr
        tot = T.sum(axis=0)
        parent_imp = float(_impurity(tot / tot_w, crit)) if crit != "twoing" else 0.0
    for j in feats:
        x = X[rows, j]
        ok = ~np.isnan(x)
        if ok.sum() < 2 * minleaf:
            continue
        order = np.argsort(x[ok], kind="mergesort")
        xs = x[ok][order]
        distinct = np.flatnonzero(xs[1:] > xs[:-1])        # split after position i
        cand = distinct[(distinct + 1 >= minleaf) & (xs.size - distinct - 1 >= minleaf)]
        if cand.size == 0:
            continue
        if K is not None:
            Ts = T[ok][order]
            cl = np.cumsum(Ts, axis=0)[cand]
            tw = Ts.sum(axis=0)
            cr = tw - cl
            wl, wrr = cl.sum(axis=1), cr.sum(axis=1)
            wt = wl + wrr
            with np.errstate(all="ignore"):
                pl, pr = cl / wl[:, None], cr / wrr[:, None]
                if crit == "twoing":
                    gain = (wl / wt) * (wrr / wt) * np.sum(np.abs(pl - pr), axis=1) ** 2 / 4
                else:
                    pimp = float(_impurity(tw / wt[0], crit)) if ok.sum() < rows.size else parent_imp
                    gain = pimp - (wl / wt) * _impurity(pl, crit) - (wrr / wt) * _impurity(pr, crit)
                gain = gain * (wt[0] / tot_w)
        else:
            ws, ys = wr[ok][order], target[rows][ok][order]
            cw, cwy = np.cumsum(ws)[cand], np.cumsum(ws * ys)[cand]
            tw, twy = ws.sum(), float(np.sum(ws * ys))
            with np.errstate(all="ignore"):
                gain = (cwy ** 2 / cw + (twy - cwy) ** 2 / (tw - cw) - twy ** 2 / tw)
        g = int(np.argmax(gain))
        scale = max(abs(float(gain[g])), 1.0)
        if gain[g] > 1e-12 * scale and (best is None or gain[g] > best[2] * (1 + 1e-12)):
            i = cand[g]
            best = (j, 0.5 * (xs[i] + xs[i + 1]), float(gain[g]))
    return best


def grow_tree(X, target, w, K, crit, minparent, minleaf, maxsplits, nvars, rng) -> Tree:
    """Grow a tree level by level. ``target`` holds class codes (K classes) or responses (K is None)."""
    n, p = X.shape
    t = Tree()
    tot_w = w.sum()

    def node_stats(rows):
        wr = w[rows]
        sw = wr.sum()
        if K is not None:
            cw = np.bincount(target[rows], weights=wr, minlength=K)
            cn = np.bincount(target[rows], minlength=K).astype(float)
            pr = cw / sw if sw > 0 else np.full(K, 1.0 / K)
            return pr, rows.size, sw / tot_w, (1 - pr.max()) * sw / tot_w, cn
        mu = float(np.sum(wr * target[rows]) / sw)
        return np.array([mu]), rows.size, sw / tot_w, float(np.sum(wr * (target[rows] - mu) ** 2) / tot_w), np.zeros(0)

    root = t.add(-1, *node_stats(np.arange(n)))
    queue = [(root, np.arange(n))]
    splits = 0
    while queue and splits < maxsplits:
        node, rows = queue.pop(0)
        if rows.size < minparent:
            continue
        if K is not None and np.count_nonzero(t.counts[node]) < 2:
            continue
        if K is None and np.ptp(target[rows]) == 0:
            continue
        feats = np.arange(p) if nvars >= p else np.sort(rng.choice(p, size=nvars, replace=False))
        best = _best_split(X, rows, w, target, K, crit, minleaf, feats)
        if best is None:
            continue
        j, cut, _ = best
        x = X[rows, j]
        lrows, rrows = rows[x < cut], rows[x >= cut]
        if lrows.size == 0 or rrows.size == 0:
            continue
        t.var[node], t.cut[node] = j, cut
        t.left[node] = t.add(node, *node_stats(lrows))
        t.right[node] = t.add(node, *node_stats(rrows))
        queue.append((t.left[node], lrows))
        queue.append((t.right[node], rrows))
        splits += 1
    return t


def merge_leaves(t: Tree) -> Tree:
    """Undo splits whose two leaves together have no lower risk than their parent."""
    collapse = np.zeros(t.n, dtype=bool)
    is_leaf = np.array([c < 0 for c in t.left])
    changed = True
    while changed:
        changed = False
        for i in range(t.n - 1, -1, -1):
            if is_leaf[i] or collapse[i]:
                continue
            a, b = t.left[i], t.right[i]
            if is_leaf[a] and is_leaf[b] and t.risk[a] + t.risk[b] >= t.risk[i] - 1e-12 * max(t.risk[i], 1e-12):
                collapse[i] = True
                is_leaf[i] = True
                changed = True
    return t.compact(collapse) if collapse.any() else t


def prune_sequence(t: Tree) -> tuple[np.ndarray, np.ndarray]:
    """Cost-complexity pruning: the level at which each branch node collapses, and the alpha of every level."""
    n = t.n
    level = np.zeros(n)
    alive = np.array([c >= 0 for c in t.left])
    alphas = [0.0]
    lvl = 0
    while alive.any():
        leaves = np.zeros(n)
        sub_risk = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if alive[i]:
                leaves[i] = leaves[t.left[i]] + leaves[t.right[i]]
                sub_risk[i] = sub_risk[t.left[i]] + sub_risk[t.right[i]]
            else:
                leaves[i], sub_risk[i] = 1, t.risk[i]
        idx = np.flatnonzero(alive)
        g = (np.array(t.risk)[idx] - sub_risk[idx]) / (leaves[idx] - 1)
        gmin = g.min()
        lvl += 1
        alphas.append(max(float(gmin), 0.0))
        for i in idx[g <= gmin + 1e-12 * max(abs(gmin), 1e-12)]:
            stack = [i]
            while stack:                      # the whole subtree goes at this level
                q = stack.pop()
                if alive[q]:
                    alive[q] = False
                    level[q] = lvl
                    stack += [t.left[q], t.right[q]]
    return level, np.array(alphas)


_TREE_OPTS = ["MinParentSize", "MinLeafSize", "MaxNumSplits", "SplitCriterion", "Prune", "MergeLeaves",
              "NumVariablesToSample", "PruneCriterion", "Surrogate", "MaxNumCategories", "AlgorithmForCategorical",
              "PredictorSelection", "Reproducible", "QuadraticErrorTolerance", "NumBins", "MinParent", "MinLeaf"]


def tree_params(o: dict, n: int, p: int, classification: bool, fname: str, defaults: dict | None = None) -> dict:
    df = {"minparent": 10, "minleaf": 1, "maxsplits": n - 1, "nvars": p, "merge": True, "crit": None}
    df.update(defaults or {})
    for bad in ("surrogate", "predictorselection"):
        if bad in o:
            v = o[bad]
            off = is_text(v) and text(v).lower() in ("off", "allsplits")
            if not off:
                raise MatlabError(f"{fname}: the option '{bad}' is not supported.")
    mp = o.get("minparentsize", o.get("minparent"))
    ml = o.get("minleafsize", o.get("minleaf"))
    if ml is not None:
        df["minleaf"] = intval(ml, "MinLeafSize")
        if mp is None:
            df["minparent"] = max(df["minparent"], 2 * df["minleaf"])
    if mp is not None:
        df["minparent"] = max(intval(mp, "MinParentSize"), 2 * df["minleaf"])
    if df["minleaf"] < 1 or df["minparent"] < 1:
        raise MatlabError(f"{fname}: MinLeafSize and MinParentSize must be positive integers.")
    if "maxnumsplits" in o:
        df["maxsplits"] = intval(o["maxnumsplits"], "MaxNumSplits")
    if "numvariablestosample" in o:
        v = o["numvariablestosample"]
        df["nvars"] = p if is_text(v) else min(intval(v, "NumVariablesToSample"), p)
    if "mergeleaves" in o:
        df["merge"] = as_bool(o["mergeleaves"], "MergeLeaves")
    if "prune" in o:
        as_bool(o["prune"], "Prune")          # the pruning sequence is computed on demand
    if classification:
        df["crit"] = choice(o["splitcriterion"], ["gdi", "deviance", "twoing"], "SplitCriterion") \
            if "splitcriterion" in o else (df["crit"] or "gdi")
    elif "splitcriterion" in o and choice(o["splitcriterion"], ["mse"], "SplitCriterion"):
        pass
    return df


class TreeModel(Model):
    posterior_scores = True
    default_loss = "mincost"

    def __init__(self, interp, d: Data, fname: str, classification: bool, params: dict | None = None,
                 tree: Tree | None = None):
        self.classification = classification
        self.matlab_class = "ClassificationTree" if classification else "RegressionTree"
        if not classification:
            self.default_loss = "mse"
        super().__init__(d, fname)
        self.interp = interp
        self.params = params or tree_params(d.opts, d.n, d.p, classification, fname)
        pr = self.params
        if tree is None:
            K = d.labels.k if classification else None
            target = d.codes if classification else d.y
            tree = grow_tree(d.X, target, self.wn, K, pr["crit"], pr["minparent"], pr["minleaf"], pr["maxsplits"],
                             pr["nvars"], rng_of(interp))
            if pr["merge"]:
                tree = merge_leaves(tree)
        self.tree = tree
        self._prune = None

    extra_props = ("NumNodes", "Children", "CutPoint", "CutPredictor", "CutPredictorIndex", "CutType",
                   "IsBranchNode", "NodeClass", "ClassProbability", "ClassCount", "NodeProbability", "NodeSize",
                   "NodeRisk", "NodeError", "NodeMean", "Parent", "PruneAlpha", "PruneList")

    def refit(self, d: Data) -> "TreeModel":
        return TreeModel(self.interp, d, self.fname, self.classification, self.params)

    def scores(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self.tree.value[i] for i in self.tree.apply(X)]) if X.shape[0] else np.zeros((0,
                self.d.labels.k))

    def response(self, X: np.ndarray) -> np.ndarray:
        vals = np.array([v[0] for v in self.tree.value])
        return vals[self.tree.apply(X)]

    def pruning(self):
        if self._prune is None:
            self._prune = prune_sequence(self.tree)
        return self._prune

    def get_field(self, interp, name: str) -> Any:
        t = self.tree
        n = t.n
        branch = np.array([c >= 0 for c in t.left])
        if name == "NumNodes":
            return float(n)
        if name == "Children":
            return np.column_stack([np.array(t.left) + 1.0, np.array(t.right) + 1.0]) * branch[:, None]
        if name == "CutPoint":
            return np.array(t.cut).reshape(-1, 1)
        if name == "CutPredictor":
            return cellstr([self.d.pnames[v] if v >= 0 else "" for v in t.var])
        if name == "CutPredictorIndex":
            return (np.array(t.var) + 1.0).reshape(-1, 1)
        if name == "CutType":
            return cellstr(["continuous" if b else "" for b in branch])
        if name == "IsBranchNode":
            return branch.reshape(-1, 1)
        if name == "Parent":
            return (np.array(t.parent) + 1.0).reshape(-1, 1)
        if name == "NodeProbability":
            return np.array(t.prob).reshape(-1, 1)
        if name == "NodeSize":
            return np.array(t.size, dtype=float).reshape(-1, 1)
        if name == "NodeRisk":
            return np.array(t.risk).reshape(-1, 1)
        if name == "NodeError":
            with np.errstate(all="ignore"):
                return (np.array(t.risk) / np.array(t.prob)).reshape(-1, 1)
        if name == "PruneList":
            return self._prune_list()
        if name == "PruneAlpha":
            return self.pruning()[1].reshape(-1, 1)
        if self.classification:
            if name == "NodeClass":
                return self.d.labels.decode(np.array([int(np.argmin(v @ self.cost)) for v in t.value]))
            if name == "ClassProbability":
                return np.vstack(t.value)
            if name == "ClassCount":
                return np.vstack(t.counts)
        elif name == "NodeMean":
            return np.array([v[0] for v in t.value]).reshape(-1, 1)
        return super().get_field(interp, name)

    def _prune_list(self) -> np.ndarray:
        """For every node, the highest pruning level at which it is still part of the tree."""
        level, _ = self.pruning()
        t = self.tree
        out = np.zeros(t.n)
        top = level.max() if level.size else 0
        for i in range(t.n):
            par = t.parent[i]
            out[i] = top if par < 0 else level[par] - 1
        return out.reshape(-1, 1)

    def m_predict(self, interp, args, nargout):
        _, opts = get_opts(args, 1, ["Subtrees"], "predict")
        if "subtrees" in opts:
            raise MatlabError("predict: the option 'Subtrees' is not supported; prune the tree first.")
        X = self.d.encode_x(args[0], "predict")
        node = self.tree.apply(X)
        if not self.classification:
            vals = np.array([v[0] for v in self.tree.value])
            return outs(nargout, vals[node].reshape(-1, 1), (node + 1.0).reshape(-1, 1))
        S = np.vstack([self.tree.value[i] for i in node]) if node.size else np.zeros((0, self.d.labels.k))
        codes = self.labels_from(S)
        return outs(nargout, self.d.labels.decode(codes), self.apply_transform(interp, S), (node + 1.0).reshape(-1, 1),
                    (codes + 1.0).reshape(-1, 1))

    def m_prune(self, interp, args, nargout):
        _, opts = get_opts(args, 0, ["Level", "Alpha", "Nodes"], "prune")
        if len(opts) > 1:
            raise MatlabError("prune: give only one of 'Level', 'Alpha' and 'Nodes'.")
        level, alphas = self.pruning()
        t = self.tree
        if "nodes" in opts:
            cut = np.zeros(t.n, dtype=bool)
            for i in vec(opts["nodes"]):
                if not 1 <= int(i) <= t.n:
                    raise MatlabError("prune: a node number is outside the tree.")
                cut[int(i) - 1] = True
        else:
            if "alpha" in opts:
                a = scalar(opts["alpha"], "Alpha")
                L = int(np.sum(alphas <= a + 1e-12 * max(abs(a), 1e-12))) - 1
            elif "level" in opts:
                L = intval(opts["level"], "Level")
            else:
                return self
            if L < 0 or L > level.max():
                raise MatlabError(f"prune: Level must be between 0 and {int(level.max())}.")
            cut = (level > 0) & (level <= L)
        return TreeModel(self.interp, self.d, self.fname, self.classification, self.params, t.compact(cut))

    def m_predictorImportance(self, interp, args, nargout):
        t = self.tree
        imp = np.zeros(self.d.p)
        nb = 0
        for i in range(t.n):
            if t.left[i] >= 0:
                nb += 1
                imp[t.var[i]] += t.risk[i] - t.risk[t.left[i]] - t.risk[t.right[i]]
        return (imp / max(nb, 1)).reshape(1, -1)

    def _node_label(self, i: int) -> str:
        if self.classification:
            return "class = " + self.d.labels.names[int(np.argmin(self.tree.value[i] @ self.cost))]
        return f"fit = {self.tree.value[i][0]:g}"

    def m_view(self, interp, args, nargout):
        _, opts = get_opts(args, 0, ["Mode"], "view")
        mode = choice(opts["mode"], ["text", "graph"], "Mode") if "mode" in opts else "text"
        t = self.tree
        if mode == "text":
            lines = [f"Decision tree for {'classification' if self.classification else 'regression'}"]
            w = len(str(t.n))
            for i in range(t.n):
                if t.left[i] >= 0:
                    nm, c = self.d.pnames[t.var[i]], f"{t.cut[i]:g}"
                    lines.append(f"{str(i + 1).rjust(w)}  if {nm}<{c} then node {t.left[i] + 1} elseif {nm}>={c} "
                                 f"then node {t.right[i] + 1} else {self._node_label(i).split('= ')[1]}")
                else:
                    lines.append(f"{str(i + 1).rjust(w)}  {self._node_label(i)}")
            interp.out("\n".join(lines) + "\n\n")
            return None
        # graph: leaves are spread along x, each parent sits above the middle of its children
        xpos, depth = np.zeros(t.n), np.zeros(t.n)
        counter = [0]

        def place(i, dep):
            depth[i] = dep
            if t.left[i] < 0:
                xpos[i] = counter[0]
                counter[0] += 1
            else:
                place(t.left[i], dep + 1)
                place(t.right[i], dep + 1)
                xpos[i] = (xpos[t.left[i]] + xpos[t.right[i]]) / 2
        place(0, 0)
        builtin_call(interp, "figure", [], 0)
        for i in range(t.n):
            if t.left[i] >= 0:
                for c in (t.left[i], t.right[i]):
                    builtin_call(interp, "plot", [np.array([[xpos[i], xpos[c]]]), np.array([[-depth[i], -depth[c]]]),
                                                  char("k-")], 0)
                    builtin_call(interp, "hold", [char("on")], 0)
                lab = f"{self.d.pnames[t.var[i]]} < {t.cut[i]:g}"
            else:
                lab = self._node_label(i).split("= ")[1]
            builtin_call(interp, "text", [np.array([[xpos[i]]]), np.array([[-depth[i]]]), char(lab),
                                          char("HorizontalAlignment"), char("center")], 0)
        builtin_call(interp, "hold", [char("off")], 0)
        builtin_call(interp, "axis", [char("off")], 0)
        return None


@builtin("fitctree")
def _fitctree(interp, args, nargout):
    d = prepare(interp, args, "fitctree", _TREE_OPTS, True, drop_nan_x=False)
    return finish(interp, TreeModel(interp, d, "fitctree", True))


@builtin("fitrtree")
def _fitrtree(interp, args, nargout):
    d = prepare(interp, args, "fitrtree", _TREE_OPTS, False, drop_nan_x=False)
    return finish(interp, TreeModel(interp, d, "fitrtree", False))


# -- naive Bayes ---------------------------------------------------------------------------------------------

class NaiveBayesModel(Model):
    matlab_class = "ClassificationNaiveBayes"
    posterior_scores = True
    default_loss = "mincost"
    extra_props = ("DistributionNames", "DistributionParameters", "Kernel", "Support", "Width")

    def __init__(self, interp, d: Data, fname: str = "fitcnb"):
        super().__init__(d, fname)
        self.interp = interp
        o = d.opts
        dn = o.get("distributionnames", char("normal"))
        names = text_list(dn) if not is_text(dn) else [text(dn)] * d.p
        if names is None or len(names) != d.p:
            raise MatlabError("fitcnb: DistributionNames must be one name or one name per predictor.")
        self.dists = [choice(char(s), ["normal", "kernel", "mn", "mvmn"], "DistributionNames") for s in names]
        if "mn" in self.dists and any(s != "mn" for s in self.dists):
            raise MatlabError("fitcnb: the multinomial ('mn') distribution must be used for all predictors.")
        kern = o.get("kernel", char("normal"))
        self.kernels = text_list(kern) if not is_text(kern) else [text(kern)] * d.p
        self.width = o.get("width")
        self.support = o.get("support")
        K = d.labels.k
        self.params: list[list[Any]] = [[None] * d.p for _ in range(K)]
        from .stats_pd import KDE, _Support
        for c in range(K):
            sel = d.codes == c
            Xc, wc = d.X[sel], self.wn[sel]
            for j, kind in enumerate(self.dists):
                x = Xc[:, j]
                ok = ~np.isnan(x)
                x, w = x[ok], wc[ok]
                if x.size == 0:
                    self.params[c][j] = None
                    continue
                w = w / w.sum()
                if kind == "normal":
                    mu = float(np.sum(w * x))
                    denom = 1 - np.sum(w ** 2)
                    sd = float(np.sqrt(np.sum(w * (x - mu) ** 2) / denom)) if denom > 0 else 0.0
                    self.params[c][j] = (mu, sd)
                elif kind == "kernel":
                    bw = None
                    if self.width is not None and not is_empty_arg(self.width):
                        W = np.asarray(fnum(self.width), dtype=float)
                        bw = float(W.flat[0]) if W.size == 1 else float(W[c, j] if W.shape == (K, d.p) else
                                                                        (W.flatten()[j] if W.size == d.p
                                                                                else W.flatten()[c]))
                    self.params[c][j] = KDE(x, w, self.kernels[j].lower(), bw, _Support(self.support, x, "fitcnb"))
                elif kind == "mvmn":
                    lv = np.unique(d.X[:, j][~np.isnan(d.X[:, j])])
                    cnt = np.array([np.sum(x == v) for v in lv], dtype=float)
                    self.params[c][j] = (lv, (1 + cnt) / (lv.size + x.size))
            if "mn" in self.dists:
                tot = np.nansum(Xc * wc[:, None], axis=0) / wc.sum() * Xc.shape[0]
                self.params[c] = [(1 + tot) / (d.p + tot.sum())] * d.p
        for c in range(K):
            for j, kind in enumerate(self.dists):
                if kind == "normal" and self.params[c][j] is not None and self.params[c][j][1] <= 0:
                    raise MatlabError(f"fitcnb: predictor '{d.pnames[j]}' has zero variance within class "
                                      f"'{d.labels.names[c]}'; a normal distribution cannot be fitted.")

    def refit(self, d: Data) -> "NaiveBayesModel":
        return NaiveBayesModel(self.interp, d, self.fname)

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        K = self.d.labels.k
        L = np.zeros((X.shape[0], K))
        with np.errstate(all="ignore"):
            for c in range(K):
                if "mn" in self.dists:
                    L[:, c] = np.nansum(X * np.log(self.params[c][0]), axis=1)
                    continue
                for j, kind in enumerate(self.dists):
                    par = self.params[c][j]
                    x = X[:, j]
                    if par is None:
                        continue
                    if kind == "normal":
                        ll = -0.5 * ((x - par[0]) / par[1]) ** 2 - np.log(par[1]) - 0.5 * np.log(2 * np.pi)
                    elif kind == "kernel":
                        ll = np.log(par.pdf(np.where(np.isnan(x), par.x[0], x)))
                    else:
                        lv, pr = par
                        idx = np.searchsorted(lv, x)
                        hit = (idx < lv.size) & (lv[np.clip(idx, 0, lv.size - 1)] == x)
                        ll = np.where(hit, np.log(pr[np.clip(idx, 0, lv.size - 1)]), -np.inf)
                    L[:, c] += np.where(np.isnan(x), 0.0, ll)
        return L

    def scores(self, X: np.ndarray) -> np.ndarray:
        from scipy.special import logsumexp
        with np.errstate(all="ignore"):
            L = self.log_likelihood(X) + np.log(self.prior)
            tot = logsumexp(L, axis=1, keepdims=True)
            P = np.exp(L - tot)
        return np.where(np.isfinite(tot), P, self.prior)

    def get_field(self, interp, name: str) -> Any:
        if name == "DistributionNames":
            return cellstr(self.dists, column=False)
        if name == "DistributionParameters":
            K, p = self.d.labels.k, self.d.p
            data = np.empty((K, p), dtype=object)
            for c in range(K):
                for j in range(p):
                    par = self.params[c][j]
                    kind = self.dists[j]
                    if par is None:
                        data[c, j] = np.zeros((0, 0))
                    elif kind == "normal":
                        data[c, j] = np.array([[par[0]], [par[1]]])
                    elif kind == "kernel":
                        data[c, j] = np.array([[par.bw]])
                    elif kind == "mvmn":
                        data[c, j] = par[1].reshape(-1, 1)
                    else:
                        data[c, j] = np.array([[float(np.asarray(par).flat[j])]])
            return Cell(data)
        if name == "Kernel":
            return cellstr(self.kernels, column=False)
        if name in ("Support", "Width"):
            v = self.support if name == "Support" else self.width
            return v if v is not None else np.zeros((0, 0))
        return super().get_field(interp, name)

    def m_logp(self, interp, args, nargout):
        from scipy.special import logsumexp
        X = self.d.encode_x(args[0], "logp")
        with np.errstate(all="ignore"):
            return logsumexp(self.log_likelihood(X) + np.log(self.prior), axis=1).reshape(-1, 1)


@builtin("fitcnb")
def _fitcnb(interp, args, nargout):
    d = prepare(interp, args, "fitcnb", ["DistributionNames", "Kernel", "Support", "Width", "Standardize"], True,
                drop_nan_x=False)
    if "standardize" in d.opts and as_bool(d.opts["standardize"], "Standardize"):
        raise MatlabError("fitcnb: the option 'Standardize' is not supported.")
    return finish(interp, NaiveBayesModel(interp, d))


# -- discriminant analysis -----------------------------------------------------------------------------------

_DISCR_TYPES = ["linear", "quadratic", "diaglinear", "diagquadratic", "pseudolinear", "pseudoquadratic"]


class DiscriminantModel(Model):
    matlab_class = "ClassificationDiscriminant"
    posterior_scores = True
    default_loss = "mincost"
    extra_props = ("DiscrimType", "Mu", "Sigma", "Gamma", "Delta", "Coeffs", "LogDetSigma", "BetweenSigma")

    def __init__(self, interp, d: Data, fname: str = "fitcdiscr", kind: str | None = None):
        super().__init__(d, fname)
        self.interp = interp
        o = d.opts
        self.kind = kind or (choice(o["discrimtype"], _DISCR_TYPES, "DiscrimType") if "discrimtype" in o else "linear")
        self.gamma = scalar(o["gamma"], "Gamma") if "gamma" in o else 0.0
        if not 0 <= self.gamma <= 1:
            raise MatlabError("fitcdiscr: Gamma must be between 0 and 1.")
        if "delta" in o and scalar(o["delta"], "Delta") != 0:
            raise MatlabError("fitcdiscr: a nonzero 'Delta' is not supported.")
        K, p = d.labels.k, d.p
        self.mu = np.full((K, p), np.nan)
        covs = []
        pooled = np.zeros((p, p))
        nused = 0
        for c in range(K):
            sel = d.codes == c
            Xc, w = d.X[sel], d.w[sel]
            if Xc.shape[0] == 0:
                covs.append(np.full((p, p), np.nan))
                continue
            wn = w / w.sum()
            self.mu[c] = wn @ Xc
            Z = Xc - self.mu[c]
            S = (Z * wn[:, None]).T @ Z
            pooled += S * w.sum()
            nused += 1
            denom = 1 - np.sum(wn ** 2)
            covs.append(S / denom if denom > 0 else np.full((p, p), np.nan))
        wsum = d.w.sum()
        eff = wsum - nused * wsum / d.n if d.n else 1.0     # (n - K) for unit weights
        pooled = pooled / eff if eff > 0 else np.full((p, p), np.nan)
        quad = "quadratic" in self.kind
        diag = self.kind.startswith("diag")
        self.pseudo = self.kind.startswith("pseudo")
        mats = covs if quad else [pooled]
        out = []
        for S in mats:
            if diag:
                S = np.diag(np.diag(S))
            elif self.gamma > 0:
                S = (1 - self.gamma) * S + self.gamma * np.diag(np.diag(S))
            out.append(S)
        self.sig = out
        self.quad = quad
        self.inv, self.logdet = [], []
        for c, S in enumerate(out):
            if np.isnan(S).any():
                self.inv.append(np.full((p, p), np.nan))
                self.logdet.append(np.nan)
                continue
            sign, ld = np.linalg.slogdet(S)
            if (sign <= 0 or np.linalg.cond(S) > 1e14) and not self.pseudo:
                which = f" of class '{d.labels.names[c]}'" if quad else ""
                raise MatlabError(f"{fname}: the covariance matrix{which} is singular. Use a 'pseudo' or 'diag' "
                                  "discriminant type or set 'Gamma'.")
            if self.pseudo:
                ev = np.linalg.eigvalsh(S)
                pos = ev[ev > ev.max() * 1e-12] if ev.size else ev
                self.inv.append(np.linalg.pinv(S))
                self.logdet.append(float(np.sum(np.log(pos))))
            else:
                self.inv.append(np.linalg.inv(S))
                self.logdet.append(float(ld))

    def refit(self, d: Data) -> "DiscriminantModel":
        return DiscriminantModel(self.interp, d, self.fname, self.kind)

    def log_dens(self, X: np.ndarray) -> np.ndarray:
        K = self.d.labels.k
        L = np.empty((X.shape[0], K))
        for c in range(K):
            i = c if self.quad else 0
            Z = X - self.mu[c]
            L[:, c] = -0.5 * np.einsum("ij,jk,ik->i", Z, self.inv[i], Z) - 0.5 * self.logdet[i] \
                - 0.5 * X.shape[1] * np.log(2 * np.pi)
        return L

    def scores(self, X: np.ndarray) -> np.ndarray:
        from scipy.special import logsumexp
        with np.errstate(all="ignore"):
            L = self.log_dens(X) + np.log(self.prior)
            L = np.where(np.isnan(L), -np.inf, L)
            return np.exp(L - logsumexp(L, axis=1, keepdims=True))

    def coeffs(self) -> Struct:
        K = self.d.labels.k
        data = np.empty((K, K), dtype=object)
        fields = ["DiscrimType", "Const", "Linear"] + (["Quadratic"] if self.quad else []) + ["Class1", "Class2"]
        cv = self.d.labels.class_value()
        for a in range(K):
            for b in range(K):
                e: dict = {f: np.zeros((0, 0)) for f in fields}
                if a != b:
                    ia, ib = (a, b) if self.quad else (0, 0)
                    ma, mb = self.mu[a], self.mu[b]
                    with np.errstate(all="ignore"):
                        const = (np.log(self.prior[a]) - np.log(self.prior[b]) - 0.5 * (self.logdet[ia]
                                - self.logdet[ib])
                                 - 0.5 * (ma @ self.inv[ia] @ ma - mb @ self.inv[ib] @ mb))
                    e.update({"DiscrimType": char(self.kind), "Const": np.array([[float(const)]]),
                              "Linear": (self.inv[ia] @ ma - self.inv[ib] @ mb).reshape(-1, 1)})
                    if self.quad:
                        e["Quadratic"] = -0.5 * (self.inv[ia] - self.inv[ib])
                    e["Class1"] = _index_label(cv, a)
                    e["Class2"] = _index_label(cv, b)
                data[a, b] = e
        return Struct(data, fields)

    def get_field(self, interp, name: str) -> Any:
        if name == "DiscrimType":
            return self.kind
        if name == "Mu":
            return self.mu
        if name == "Sigma":
            S = np.stack(self.sig, axis=2) if self.quad else self.sig[0]
            if self.kind.startswith("diag"):
                return np.stack([np.diag(s).reshape(1, -1) for s in self.sig], axis=2) if self.quad \
                    else np.diag(self.sig[0]).reshape(1, -1)
            return S
        if name == "Gamma":
            return float(self.gamma)
        if name == "Delta":
            return 0.0
        if name == "Coeffs":
            return self.coeffs()
        if name == "LogDetSigma":
            return np.array(self.logdet).reshape(-1, 1) if self.quad else float(self.logdet[0])
        if name == "BetweenSigma":
            gm = self.prior @ np.nan_to_num(self.mu)
            Z = np.nan_to_num(self.mu) - gm
            return (Z * self.prior[:, None]).T @ Z
        return super().get_field(interp, name)

    def m_mahal(self, interp, args, nargout):
        X = self.d.encode_x(args[0], "mahal")
        K = self.d.labels.k
        out = np.empty((X.shape[0], K))
        for c in range(K):
            Z = X - self.mu[c]
            out[:, c] = np.einsum("ij,jk,ik->i", Z, self.inv[c if self.quad else 0], Z)
        return out


def _index_label(cv: Any, i: int) -> Any:
    if isinstance(cv, Cell):
        return cv.data.flat[i]
    if isinstance(cv, np.ndarray):
        return cv[i:i + 1]
    if hasattr(cv, "data") and hasattr(cv, "like"):
        return cv.like(cv.data[i:i + 1])
    return cv


@builtin("fitcdiscr")
def _fitcdiscr(interp, args, nargout):
    d = prepare(interp, args, "fitcdiscr", ["DiscrimType", "Gamma", "Delta", "FillCoeffs", "SaveMemory"], True)
    return finish(interp, DiscriminantModel(interp, d))


@builtin("classify")
def _classify(interp, args, nargout):
    if not 3 <= len(args) <= 5:
        raise MatlabError("classify: expected classify(sample, training, group, type, prior).")
    sample = np.asarray(fnum(args[0]), dtype=float)
    kind = choice(args[3], ["linear", "diaglinear", "quadratic", "diagquadratic", "mahalanobis"], "The type") \
        if given(args, 3) else "linear"
    extra = [args[1], args[2]]
    if given(args, 4):
        v = args[4]
        extra += [char("Prior"), char("empirical") if is_text(v) and text(v).lower() == "empirical" else v]
    d = prepare(interp, extra, "classify", [], True)
    if sample.ndim != 2 or sample.shape[1] != d.p:
        raise MatlabError("classify: sample and training must have the same number of columns.")
    if not given(args, 4):
        d.opts["prior"] = char("uniform")
    m = DiscriminantModel(interp, d, "classify", "quadratic" if kind == "mahalanobis" else kind)
    if kind == "mahalanobis":
        dist = m.m_mahal(interp, [sample], 1)
        codes = np.argmin(dist, axis=1)
        post, logp = np.full(dist.shape, np.nan), np.full((sample.shape[0], 1), np.nan)
        train_codes = np.argmin(m.m_mahal(interp, [d.X], 1), axis=1)
    else:
        from scipy.special import logsumexp
        with np.errstate(all="ignore"):
            L = m.log_dens(sample) + np.log(m.prior)
            tot = logsumexp(L, axis=1, keepdims=True)
        post, logp = np.exp(L - tot), tot
        codes = np.argmax(post, axis=1)
        train_codes = np.argmax(m.scores(d.X), axis=1)
    err = float(np.sum(m.prior * np.array([np.mean(train_codes[d.codes == c] != c) if np.any(d.codes == c) else 0.0
                                           for c in range(d.labels.k)])))
    return outs(nargout, d.labels.decode(codes), err, post, logp, m.coeffs())


# -- assessment tools ----------------------------------------------------------------------------------------

@builtin("confusionmat")
def _confusionmat(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("confusionmat: expected confusionmat(group, grouphat).")
    pos, opts = get_opts(args, 2, ["Order"], "confusionmat")
    if len(pos) != 2:
        raise MatlabError("confusionmat: expected confusionmat(group, grouphat, 'Order', order).")
    a, b = Labels(pos[0]), Labels(pos[1])
    if a.n != b.n:
        raise MatlabError("confusionmat: group and grouphat must have the same number of elements.")
    if (a.kind in ("numeric", "logical")) != (b.kind in ("numeric", "logical")):
        raise MatlabError("confusionmat: group and grouphat must have the same type.")
    if "order" in opts:
        ref = Labels(opts["order"], opts["order"])
    else:
        merged = sorted(set(a.classes) | set(b.classes)) if a.kind != "categorical" else \
            [c for c in (a.cats or []) if c in set(a.classes) | set(b.classes)] + \
            [c for c in b.classes if c not in (a.cats or [])]
        ref = Labels(pos[0])
        ref.classes = merged
    lookup = {k: i for i, k in enumerate(ref.classes)}
    K = len(ref.classes)
    C = np.zeros((K, K))
    for ca, cb in zip(a.codes, b.codes):
        if ca < 0 or cb < 0:
            continue
        i, j = lookup.get(a.classes[ca]), lookup.get(b.classes[cb])
        if i is not None and j is not None:
            C[i, j] += 1
    return outs(nargout, C, ref.decode(np.arange(K)))


class ConfusionChart(StatObject):
    matlab_class = "matlab.graphics.chart.ConfusionMatrixChart"


@builtin("confusionchart")
def _confusionchart(interp, args, nargout):
    if not args:
        raise MatlabError("confusionchart: expected confusionchart(trueLabels, predictedLabels) or confusionchart(m).")
    known = ["Normalization", "Title", "XLabel", "YLabel", "RowSummary", "ColumnSummary", "Order", "FontSize"]
    pos, opts = get_opts(args, 1, known, "confusionchart")
    for o in ("rowsummary", "columnsummary"):
        if o in opts and text(opts[o]).lower() != "off":
            raise MatlabError(f"confusionchart: the option '{o}' is not supported.")
    first = np.asarray(fnum(pos[0])) if isinstance(pos[0], np.ndarray) and pos[0].dtype.kind in "biuf" else None
    square = first is not None and first.ndim == 2 and first.shape[0] == first.shape[1] and first.shape[0] > 1
    if square and len(pos) <= 2:
        C = first.astype(float)
        names = text_list(pos[1]) if len(pos) > 1 and text_list(pos[1]) is not None else \
            ([str(v) for v in Labels(pos[1]).names] if len(pos) > 1 else [str(i + 1) for i in range(C.shape[0])])
    else:
        if len(pos) != 2:
            raise MatlabError("confusionchart: expected confusionchart(trueLabels, predictedLabels).")
        extra = [char("Order"), opts["order"]] if "order" in opts else []
        C, order = _confusionmat(interp, [pos[0], pos[1]] + extra, 2)
        names = Labels(order, order).names
    norm = choice(opts["normalization"], ["absolute", "row-normalized", "column-normalized", "total-normalized"],
                  "Normalization") if "normalization" in opts else "absolute"
    with np.errstate(all="ignore"):
        V = {"absolute": C, "row-normalized": C / C.sum(axis=1, keepdims=True),
             "column-normalized": C / C.sum(axis=0, keepdims=True), "total-normalized": C / C.sum()}[norm]
    K = C.shape[0]
    builtin_call(interp, "imagesc", [V], 0)
    for i in range(K):
        for j in range(K):
            lab = f"{int(C[i, j])}" if norm == "absolute" else f"{100 * V[i, j]:.1f}%"
            builtin_call(interp, "text", [np.array([[j + 1.0]]), np.array([[i + 1.0]]), char(lab),
                                          char("HorizontalAlignment"), char("center")], 0)
    gca = builtin_call(interp, "gca", [], 1)[0]
    ticks = np.arange(1.0, K + 1).reshape(1, -1)
    builtin_call(interp, "set", [gca, char("XTick"), ticks, char("XTickLabel"), cellstr(names, column=False),
                                 char("YTick"), ticks, char("YTickLabel"), cellstr(names, column=False)], 0)
    builtin_call(interp, "xlabel", [opts.get("xlabel", char("Predicted Class"))], 0)
    builtin_call(interp, "ylabel", [opts.get("ylabel", char("True Class"))], 0)
    if "title" in opts:
        builtin_call(interp, "title", [opts["title"]], 0)
    chart = ConfusionChart()
    chart.p = {"NormalizedValues": V, "ClassLabels": cellstr(names), "Normalization": norm,
               "Title": opts.get("title", char(""))}
    return chart if nargout else None


_PERF_CRIT = ["fpr", "tpr", "fnr", "tnr", "ppv", "npv", "accu", "tp", "fn", "fp", "tn", "reca", "sens", "spec",
              "prec", "fall", "miss", "ecost", "rpp", "rnp"]


def _perf_value(name: str, tp, fp, fn, tn, cost):
    with np.errstate(all="ignore"):
        P, N = tp + fn, fp + tn
        table = {"fpr": fp / N, "fall": fp / N, "tpr": tp / P, "reca": tp / P, "sens": tp / P, "fnr": fn / P,
                 "miss": fn / P, "tnr": tn / N, "spec": tn / N, "ppv": tp / (tp + fp), "prec": tp / (tp + fp),
                 "npv": tn / (tn + fn), "accu": (tp + tn) / (P + N), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                 "rpp": (tp + fp) / (P + N), "rnp": (tn + fn) / (P + N),
                 "ecost": (tp * cost[0, 0] + fn * cost[0, 1] + fp * cost[1, 0] + tn * cost[1, 1]) / (P + N)}
        return table[name]


@builtin("perfcurve")
def _perfcurve(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("perfcurve: expected perfcurve(labels, scores, posclass, 'Name', value, ...).")
    known = ["NegClass", "XCrit", "YCrit", "XVals", "TVals", "UseNearest", "ProcessNaN", "Prior", "Cost", "Weights",
             "Alpha", "NBoot", "BootType", "BootArg", "Options"]
    pos, opts = get_opts(args, 3, known, "perfcurve")
    for o in ("nboot", "boottype", "bootarg", "xvals", "tvals", "prior"):
        if o in opts:
            raise MatlabError(f"perfcurve: the option '{o}' is not supported.")
    lab = Labels(pos[0])
    s = vec(pos[1])
    if lab.n != s.size:
        raise MatlabError("perfcurve: labels and scores must have the same number of elements.")
    pc = lab.encode(pos[2])
    if pc.size != 1 or pc[0] < 0:
        raise MatlabError("perfcurve: the positive class is not one of the labels.")
    w = vec(opts["weights"]) if "weights" in opts else np.ones(s.size)
    neg_ok = np.ones(lab.k, dtype=bool)
    if "negclass" in opts and not (is_text(opts["negclass"]) and text(opts["negclass"]).lower() == "all"):
        neg_ok[:] = False
        neg_ok[lab.encode(opts["negclass"])] = True
        neg_ok[pc[0]] = False
    nan_mode = choice(opts["processnan"], ["ignore", "addtofalse"], "ProcessNaN") if "processnan" in opts else "ignore"
    is_pos = lab.codes == pc[0]
    use = (lab.codes >= 0) & (is_pos | neg_ok[np.clip(lab.codes, 0, None)])
    if nan_mode == "ignore":
        use &= ~np.isnan(s)
    else:
        s = np.where(np.isnan(s), -np.inf, s)
    s, is_pos, w = s[use], is_pos[use], w[use]
    cost = np.asarray(fnum(opts["cost"]), dtype=float) if "cost" in opts else np.array([[0.0, 1.0], [1.0, 0.0]])
    if cost.shape != (2, 2):
        raise MatlabError("perfcurve: Cost must be a 2-by-2 matrix [0 C(N|P); C(P|N) 0].")
    order = np.argsort(-s, kind="mergesort")
    s, is_pos, w = s[order], is_pos[order], w[order]
    P, N = float(np.sum(w[is_pos])), float(np.sum(w[~is_pos]))
    if P == 0 or N == 0:
        raise MatlabError("perfcurve: both the positive and the negative class need at least one observation.")
    last = np.flatnonzero(np.concatenate([s[1:] != s[:-1], [True]]))     # last index of each distinct score
    tp = np.concatenate([[0.0], np.cumsum(w * is_pos)[last]])
    fp = np.concatenate([[0.0], np.cumsum(w * ~is_pos)[last]])
    T = np.concatenate([[s[0]], s[last]])
    fn, tn = P - tp, N - fp
    xc = choice(opts["xcrit"], _PERF_CRIT, "XCrit") if "xcrit" in opts else "fpr"
    yc = choice(opts["ycrit"], _PERF_CRIT, "YCrit") if "ycrit" in opts else "tpr"
    X = _perf_value(xc, tp, fp, fn, tn, cost)
    Y = _perf_value(yc, tp, fp, fn, tn, cost)
    ok = ~(np.isnan(X) | np.isnan(Y))
    xs, ys = X[ok], Y[ok]
    auc = float(abs(np.sum(np.diff(xs) * (ys[1:] + ys[:-1]) / 2))) if ok.sum() > 1 else np.nan
    with np.errstate(all="ignore"):
        slope = (cost[1, 0] - cost[1, 1]) / (cost[0, 1] - cost[0, 0]) * N / P
    fpr, tpr = fp / N, tp / P
    best = int(np.argmax(tpr - slope * fpr))
    opt = np.array([[X[best], Y[best]]]) if xc == "fpr" and yc == "tpr" else np.array([[fpr[best], tpr[best]]])
    return outs(nargout, X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1), auc, opt)


# -- learner templates ---------------------------------------------------------------------------------------

class Template(StatObject):
    def __init__(self, kind: str, args: list):
        super().__init__()
        self.kind = kind
        self.args = list(args)
        self.matlab_class = {"tree": "classreg.learning.FitTemplate", }.get(kind, "classreg.learning.FitTemplate")

    def display_lines(self) -> list[str]:
        return [f"Fit template for {self.kind}."]


def _template(kind: str):
    def f(interp, args, nargout):
        if len(args) % 2:
            raise MatlabError(f"template{kind}: options are given as name-value pairs.")
        return Template(kind.lower(), list(args))
    return f


for _k in ("Tree", "SVM", "KNN", "Discriminant", "NaiveBayes", "Linear"):
    builtin(f"template{_k}")(_template(_k))


def learner_spec(v: Any, fname: str) -> tuple[str, list]:
    """Kind and option list of a 'Learners' value (a name or a template object)."""
    if isinstance(v, Template):
        return v.kind, list(v.args)
    if is_text(v):
        s = text(v).lower()
        kinds = {"svm": "svm", "tree": "tree", "knn": "knn", "discriminant": "discriminant", "naivebayes": "naivebayes",
                 "linear": "linear"}
        if s in kinds:
            return kinds[s], []
    raise MatlabError(f"{fname}: Learners must be a learner name or a template object.")


# -- feature selection ---------------------------------------------------------------------------------------

@builtin("sequentialfs")
def _sequentialfs(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("sequentialfs: expected sequentialfs(fun, X, y, 'Name', value, ...).")
    known = ["cv", "mcreps", "direction", "keepin", "keepout", "nfeatures", "nullmodel", "options"]
    pos, opts = get_opts(args, 2, known, "sequentialfs")
    fun, data = pos[0], pos[1:]
    X = np.asarray(fnum(data[0]), dtype=float)
    n, p = X.shape
    others = data[1:]
    from .stats_desc import CVPartition, _take, make_partition
    cv = opts.get("cv")
    if cv is None:
        part = make_partition(rng_of(interp), n, "kfold", float(min(10, n)), None).train_test()
    elif isinstance(cv, CVPartition):
        part = cv.train_test()
    elif is_text(cv):
        if text(cv).lower() != "none":
            raise MatlabError("sequentialfs: cv must be a number of folds, a cvpartition or 'none'.")
        part = None
    else:
        part = make_partition(rng_of(interp), n, "kfold", intval(cv, "cv"), None).train_test()
    forward = not ("direction" in opts and choice(opts["direction"], ["forward", "backward"],
            "direction") == "backward")
    keepin = np.zeros(p, dtype=bool)
    keepout = np.zeros(p, dtype=bool)
    for key, mask in (("keepin", keepin), ("keepout", keepout)):
        if key in opts:
            v = np.asarray(opts[key])
            if v.dtype == bool:
                mask[:] = v.flatten()
            else:
                mask[vec(opts[key]).astype(int) - 1] = True
    nfeat = intval(opts["nfeatures"], "nfeatures") if "nfeatures" in opts else None

    def crit(mask: np.ndarray) -> float:
        Xs = X[:, mask]
        if part is None:
            return float(fnum(call(interp, fun, [Xs] + list(others), 1)[0]).flat[0])
        tot = 0.0
        for tr, te in part:
            a = [Xs[tr]] + [_take(o, tr) for o in others] + [Xs[te]] + [_take(o, te) for o in others]
            tot += float(fnum(call(interp, fun, a, 1)[0]).flat[0])
        return tot / n

    cur = keepin.copy() if forward else ~keepout
    best = crit(cur) if cur.any() else np.inf
    hist_in, hist_crit = [], []
    while True:
        cand = np.flatnonzero(~cur & ~keepout) if forward else np.flatnonzero(cur & ~keepin)
        if cand.size == 0 or (nfeat is not None and int(cur.sum()) == nfeat):
            break
        vals = []
        for j in cand:
            m = cur.copy()
            m[j] = forward
            vals.append(crit(m) if m.any() else np.inf)
        k = int(np.argmin(vals))
        improved = vals[k] < best - 1e-6 * abs(best) if np.isfinite(best) else True
        if nfeat is None and not improved:
            break
        cur[cand[k]] = forward
        best = vals[k]
        hist_in.append(cur.copy())
        hist_crit.append(best)
    hist = struct(In=np.array(hist_in, dtype=bool).reshape(len(hist_in), p), Crit=np.array([hist_crit]))
    return outs(nargout, cur.reshape(1, -1), hist)


@builtin("relieff")
def _relieff(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("relieff: expected relieff(X, y, k, 'Name', value, ...).")
    pos, opts = get_opts(args, 3, ["method", "prior", "updates", "categoricalx", "sigma"], "relieff")
    for o in ("prior", "categoricalx", "sigma"):
        if o in opts and not (is_text(opts[o]) and text(opts[o]).lower() == "off"):
            raise MatlabError(f"relieff: the option '{o}' is not supported.")
    X = np.asarray(fnum(pos[0]), dtype=float)
    k = intval(pos[2], "k")
    n, p = X.shape
    yv = pos[1]
    numeric_y = isinstance(yv, np.ndarray) and yv.dtype.kind in "iuf"
    method = choice(opts["method"], ["classification", "regression"], "method") if "method" in opts else \
        ("regression" if numeric_y else "classification")
    span = X.max(axis=0) - X.min(axis=0)
    span = np.where(span > 0, span, 1.0)
    Z = (X - X.min(axis=0)) / span
    D = np.abs(Z[:, None, :] - Z[None, :, :]).sum(axis=2)
    np.fill_diagonal(D, np.inf)
    updates = n
    if "updates" in opts and not is_text(opts["updates"]):
        updates = min(intval(opts["updates"], "updates"), n)
    W = np.zeros(p)
    if method == "classification":
        lab = Labels(yv)
        codes = lab.codes
        prior = np.array([np.mean(codes == c) for c in range(lab.k)])
        for i in range(updates):
            for c in range(lab.k):
                members = np.flatnonzero((codes == c) & (np.arange(n) != i))
                if members.size == 0:
                    continue
                near = members[np.argsort(D[i, members], kind="stable")[:k]]
                diff = np.abs(Z[i] - Z[near]).mean(axis=0)
                if c == codes[i]:
                    W -= diff / updates
                else:
                    W += prior[c] / (1 - prior[codes[i]]) * diff / updates
    else:
        y = vec(yv)
        yr = (y - y.min()) / ((y.max() - y.min()) or 1.0)
        ndy, ndf, ndyf = 0.0, np.zeros(p), np.zeros(p)
        for i in range(updates):
            near = np.argsort(D[i], kind="stable")[:k]
            dy = np.abs(yr[i] - yr[near])
            df = np.abs(Z[i] - Z[near])
            ndy += dy.mean()
            ndf += df.mean(axis=0)
            ndyf += (dy[:, None] * df).mean(axis=0)
        with np.errstate(all="ignore"):
            W = ndyf / ndy - (ndf - ndyf) / (updates - ndy)
    ranked = np.argsort(-W, kind="stable") + 1.0
    return outs(nargout, ranked.reshape(1, -1), W.reshape(1, -1))
