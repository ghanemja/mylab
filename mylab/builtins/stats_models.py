"""Linear and generalized linear models: Wilkinson formulas, design matrices, fitlm, stepwiselm,
fitglm, glmfit and glmval."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct, char
from ._registry import builtin
from ._stats_util import (
    StatObject, as_bool, builtin_call, cellstr, choice, get_opts, group_codes, intval, is_categorical,
    is_empty_arg, is_table, make_table, num_text, outs, rng_of, scalar, struct, text_list, vec,
)
from ._util import fnum, is_text, text


# -- model data ------------------------------------------------------------------------------------------

class ModelData:
    """Variables of a regression problem: numeric columns and categorical codes with their levels."""

    def __init__(self):
        self.names: list[str] = []
        self.values: list[np.ndarray] = []       # float column, or integer codes (-1 = missing)
        self.levels: list[list[str] | None] = []  # None for numeric variables
        self.resp: int | None = None
        self.preds: list[int] = []
        self.n = 0

    def add(self, name: str, v: Any, categorical: bool = False) -> None:
        if categorical or is_categorical(v) or text_list(v) is not None or (
                isinstance(v, np.ndarray) and v.dtype == bool and categorical):
            codes, names, _ = group_codes(v)
            self.values.append(codes)
            self.levels.append(list(names))
        else:
            a = np.asarray(fnum(v), dtype=float)
            if a.ndim == 2 and a.shape[1] != 1 and a.shape[0] != 1:
                raise MatlabError(f"The variable '{name}' must be a column vector.")
            self.values.append(a.flatten(order="F"))
            self.levels.append(None)
        self.names.append(name)
        self.n = self.values[-1].size

    def missing(self, idx: list[int]) -> np.ndarray:
        bad = np.zeros(self.n, dtype=bool)
        for j in idx:
            bad |= (self.values[j] < 0) if self.levels[j] is not None else np.isnan(self.values[j])
        return bad


def _names_arg(v: Any, what: str) -> list[str]:
    items = text_list(v)
    if items is None:
        raise MatlabError(f"{what} must be a cell array of names.")
    return items


def _var_select(v: Any, names: list[str], what: str) -> list[int]:
    """Indices of variables given by name(s), numbers or a logical mask."""
    items = text_list(v)
    if items is not None:
        out = []
        for s in items:
            if s not in names:
                raise MatlabError(f"{what}: '{s}' is not a variable name.")
            out.append(names.index(s))
        return out
    a = np.asarray(v)
    if a.dtype == bool:
        return [int(i) for i in np.flatnonzero(a.flatten())]
    return [int(i) - 1 for i in vec(v)]


def model_data(pos: list, opts: dict, fname: str) -> tuple[ModelData, list]:
    """Read (tbl, ...) or (X, y, ...) into a ModelData; returns it and the remaining positional args."""
    md = ModelData()
    if not pos:
        raise MatlabError(f"Not enough input arguments for {fname}.")
    if is_table(pos[0]):
        tbl = pos[0]
        rest = list(pos[1:])
        cat = set(_var_select(opts["categoricalvars"], tbl.names, "CategoricalVars")) \
            if "categoricalvars" in opts else set()
        for j, (nm, c) in enumerate(zip(tbl.names, tbl.columns)):
            is_cat = j in cat or is_categorical(c) or text_list(c) is not None or \
                (isinstance(c, np.ndarray) and c.dtype == bool)
            if not is_cat and not (isinstance(c, np.ndarray) and c.dtype.kind in "iuf" and (c.ndim < 2
                    or c.shape[1] == 1)):
                md.names.append(nm)       # unusable type: kept as a placeholder that cannot enter a model
                md.values.append(np.full(tbl.nrows, np.nan))
                md.levels.append(None)
                md.n = tbl.nrows
                continue
            md.add(nm, c, is_cat)
        md.n = tbl.nrows
        md.resp = len(md.names) - 1
        if "responsevar" in opts:
            md.resp = _var_select(opts["responsevar"], md.names, "ResponseVar")[0]
        md.preds = [j for j in range(len(md.names)) if j != md.resp]
        if "predictorvars" in opts:
            md.preds = _var_select(opts["predictorvars"], md.names, "PredictorVars")
        if "varnames" in opts:
            raise MatlabError(f"{fname}: VarNames applies to matrix input only.")
        return md, rest
    if len(pos) < 2:
        raise MatlabError(f"{fname}: expected {fname}(X, y, ...) or {fname}(tbl, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError(f"{fname}: X must be a matrix with one observation per row.")
    y_arg = pos[1]
    p = X.shape[1]
    names = [f"x{j + 1}" for j in range(p)] + ["y"]
    if "varnames" in opts:
        names = _names_arg(opts["varnames"], "VarNames")
        if len(names) != p + 1:
            raise MatlabError(f"{fname}: VarNames must list {p + 1} names (the predictors and the response).")
    if "predictorvars" in opts and text_list(opts["predictorvars"]) is not None and "varnames" not in opts:
        pn = _names_arg(opts["predictorvars"], "PredictorVars")
        if len(pn) == p:
            names = pn + [names[-1]]
            opts = {k: v for k, v in opts.items() if k != "predictorvars"}
    if "responsevar" in opts and is_text(opts["responsevar"]):
        names[-1] = text(opts["responsevar"])
    cat = set(_var_select(opts["categoricalvars"], names, "CategoricalVars")) if "categoricalvars" in opts else set()
    for j in range(p):
        md.add(names[j], X[:, j].reshape(-1, 1), j in cat)
    yv = y_arg
    if isinstance(y_arg, np.ndarray) and y_arg.ndim == 2 and y_arg.shape[0] == 1 and y_arg.shape[1] == X.shape[0]:
        yv = y_arg.reshape(-1, 1)
    if isinstance(yv, np.ndarray) and yv.dtype.kind in "biuf" and yv.ndim == 2 and yv.shape[1] == 2 and \
            yv.shape[0] == X.shape[0]:
        md.add(names[-1], yv[:, 0].reshape(-1, 1))   # [successes trials]: the trials are read by fitglm
        md.trials = np.asarray(yv[:, 1], dtype=float)
    else:
        md.add(names[-1], yv)
    if md.values[-1].size != X.shape[0]:
        raise MatlabError(f"{fname}: y must have one value per row of X.")
    md.n = X.shape[0]
    md.resp = p
    md.preds = list(range(p))
    if "predictorvars" in opts:
        md.preds = _var_select(opts["predictorvars"], names, "PredictorVars")
    return md, list(pos[2:])


# -- Wilkinson formulas ----------------------------------------------------------------------------------------

Term = tuple  # powers of every variable, e.g. (1, 0, 2)

_TOKEN = re.compile(r"\s*([A-Za-z_]\w*|\d+|[-+*:^()~])")


def _tokens(s: str) -> list[str]:
    out, i = [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = _TOKEN.match(s, i)
        if not m:
            raise MatlabError(f"Cannot understand the formula near '{s[i:]}'.")
        out.append(m.group(1))
        i = m.end()
    return out


class _FormulaParser:
    def __init__(self, toks: list[str], names: list[str]):
        self.toks = toks
        self.i = 0
        self.names = names
        self.nv = len(names)
        self.no_intercept = False

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def zero(self) -> Term:
        return tuple([0] * self.nv)

    @staticmethod
    def union(a: list, b: list) -> list:
        return a + [t for t in b if t not in a]

    @staticmethod
    def product(a: list, b: list) -> list:
        out = []
        for s in a:
            for t in b:
                u = tuple(x + y for x, y in zip(s, t))
                if u not in out:
                    out.append(u)
        return out

    def expr(self) -> list:
        terms: list = []
        sign = "+"
        if self.peek() in ("+", "-"):
            sign = self.take()
        while True:
            grp = self.cross()
            if sign == "+":
                terms = self.union(terms, grp)
            else:
                if self.zero() in grp:
                    self.no_intercept = True
                terms = [t for t in terms if t not in grp]
            if self.peek() in ("+", "-"):
                sign = self.take()
            else:
                return terms

    def cross(self) -> list:
        left = self.inter()
        while self.peek() == "*":
            self.take()
            right = self.inter()
            left = self.union(self.union(left, right), self.product(left, right))
        return left

    def inter(self) -> list:
        left = self.atom()
        while self.peek() == ":":
            self.take()
            left = self.product(left, self.atom())
        return left

    def atom(self) -> list:
        t = self.take()
        if t is None:
            raise MatlabError("The formula ends unexpectedly.")
        if t == "(":
            inner = self.expr()
            if self.take() != ")":
                raise MatlabError("Unbalanced parentheses in the formula.")
            if self.peek() == "^":
                self.take()
                k = int(self.take())
                res = list(inner)
                for _ in range(k - 1):
                    res = self.union(self.union(res, inner), [u for u in self.product(res, inner) if max(u) <= 1])
                return res
            return inner
        if t.isdigit():
            if t == "1":
                return [self.zero()]
            if t == "0":
                self.no_intercept = True
                return []
            raise MatlabError(f"Unexpected number '{t}' in the formula.")
        if t not in self.names:
            raise MatlabError(f"The formula uses '{t}', which is not a variable.")
        j = self.names.index(t)
        k = 1
        if self.peek() == "^":
            self.take()
            k = int(self.take())
        out = []
        for q in range(1, k + 1):   # x^k stands for all powers up to k
            u = [0] * self.nv
            u[j] = q
            out.append(tuple(u))
        return out


def _sort_terms(terms: list) -> list:
    keyed = [(sum(t), max(t) if t else 0, i, t) for i, t in enumerate(terms)]
    return [k[3] for k in sorted(keyed, key=lambda k: (k[0], k[1], k[2]))]


def parse_formula(formula: str, names: list[str]) -> tuple[str, list, bool]:
    """Response name, list of terms (power tuples, without the intercept) and whether an intercept is present."""
    if "~" not in formula:
        raise MatlabError("A formula has the form 'response ~ terms'.")
    lhs, rhs = formula.split("~", 1)
    resp = lhs.strip()
    if resp not in names:
        raise MatlabError(f"The formula response '{resp}' is not a variable.")
    p = _FormulaParser(_tokens(rhs), names)
    terms = p.expr()
    if p.peek() is not None:
        raise MatlabError(f"Cannot understand the formula near '{p.peek()}'.")
    zero = p.zero()
    intercept = not p.no_intercept
    terms = [t for t in terms if t != zero]
    return resp, _sort_terms(terms), intercept


def spec_terms(spec: Any, md: ModelData) -> tuple[list, bool]:
    """Terms for a model specification name ('linear', 'quadratic', ...), a terms matrix or a formula."""
    nv = len(md.names)

    def unit(j, k=1):
        u = [0] * nv
        u[j] = k
        return tuple(u)

    numeric = [j for j in md.preds if md.levels[j] is None]
    if spec is None or is_empty_arg(spec):
        spec = char("linear")
    if is_text(spec):
        s = text(spec).strip()
        if "~" in s:
            resp, terms, icpt = parse_formula(s, md.names)
            md.resp = md.names.index(resp)
            used = sorted({j for t in terms for j in range(nv) if t[j]})
            if md.resp in used:
                raise MatlabError("The response cannot also be a predictor in the formula.")
            md.preds = [j for j in range(nv) if j != md.resp and (j in used or j in md.preds)]
            md.preds = [j for j in md.preds if j != md.resp]
            return terms, icpt
        low = s.lower()
        mains = [unit(j) for j in md.preds]
        inter = [tuple(a + b for a, b in zip(mains[i], mains[k])) for i in range(len(mains))
                 for k in range(i + 1, len(mains))]
        squares = [unit(j, 2) for j in numeric]
        if low == "constant":
            return [], True
        if low == "linear":
            return mains, True
        if low == "interactions":
            return mains + inter, True
        if low == "purequadratic":
            return mains + squares, True
        if low == "quadratic":
            return mains + inter + squares, True
        m = re.fullmatch(r"poly(\d+)", low)
        if m:
            degs = [int(c) for c in m.group(1)]
            if len(degs) != len(md.preds):
                raise MatlabError("A 'polyIJK' model needs one digit per predictor.")
            from itertools import product
            terms = []
            for combo in product(*[range(d + 1) for d in degs]):
                if sum(combo) == 0 or sum(combo) > max(degs):
                    continue
                u = [0] * nv
                for j, k in zip(md.preds, combo):
                    u[j] = k
                terms.append(tuple(u))
            return _sort_terms(terms), True
        raise MatlabError(f"Unknown model specification '{s}'.")
    T = np.atleast_2d(np.asarray(fnum(spec))).astype(int)
    if T.shape[1] == len(md.preds) and T.shape[1] != nv:
        full = np.zeros((T.shape[0], nv), dtype=int)
        full[:, md.preds] = T
        T = full
    if T.shape[1] != nv:
        raise MatlabError(f"A terms matrix must have {nv} columns (one per variable, the response included).")
    terms = [tuple(int(v) for v in r) for r in T]
    zero = tuple([0] * nv)
    return [t for t in terms if t != zero], zero in terms


def term_name(t: Term, names: list[str]) -> str:
    parts = [names[j] if k == 1 else f"{names[j]}^{k}" for j, k in enumerate(t) if k]
    return ":".join(parts) if parts else "1"


def design(md: ModelData, terms: list, intercept: bool, values: list | None = None) -> tuple[np.ndarray, list, list]:
    """Design matrix, coefficient names and the term index of every column (-1 = intercept)."""
    vals = md.values if values is None else values
    n = vals[md.preds[0]].size if md.preds else (len(vals[0]) if vals else 0)
    for v in vals:
        if v is not None:
            n = len(v)
            break
    cols, names, owner = [], [], []
    if intercept:
        cols.append(np.ones(n))
        names.append("(Intercept)")
        owner.append(-1)
    for ti, t in enumerate(terms):
        block = [np.ones(n)]
        bnames = [""]
        for j, k in enumerate(t):
            if not k:
                continue
            if md.levels[j] is None:
                x = np.asarray(vals[j], dtype=float) ** k
                block = [b * x for b in block]
                lab = md.names[j] if k == 1 else f"{md.names[j]}^{k}"
                bnames = [f"{b}:{lab}" if b else lab for b in bnames]
            else:
                if k != 1:
                    raise MatlabError(f"The categorical variable '{md.names[j]}' cannot be raised to a power.")
                lv = md.levels[j]
                codes = np.asarray(vals[j])
                nb, nn = [], []
                for b, bn in zip(block, bnames):
                    for q in range(1, len(lv)):
                        nb.append(b * (codes == q))
                        lab = f"{md.names[j]}_{lv[q]}"
                        nn.append(f"{bn}:{lab}" if bn else lab)
                block, bnames = nb, nn
        cols.extend(block)
        names.extend(bnames)
        owner.extend([ti] * len(block))
    X = np.column_stack(cols) if cols else np.zeros((n, 0))
    return X, names, owner


def formula_text(md: ModelData, terms: list, intercept: bool, link: str | None = None) -> str:
    parts = (["1"] if intercept else []) + [term_name(t, md.names) for t in terms]
    lhs = md.names[md.resp]
    if link and link != "identity":
        lhs = f"{link}({lhs})"
    return f"{lhs} ~ " + (" + ".join(parts) if parts else "0")


def new_values(md: ModelData, Xnew: Any, fname: str) -> list:
    """Values of the model variables for new data (a matrix with the predictor columns, or a table)."""
    vals: list = [None] * len(md.names)
    if is_table(Xnew):
        cols = dict(zip(Xnew.names, Xnew.columns))
        for j in md.preds:
            if md.names[j] not in cols:
                raise MatlabError(f"{fname}: the table has no variable '{md.names[j]}'.")
            vals[j] = _encode(md, j, cols[md.names[j]])
        return vals
    X = np.asarray(fnum(Xnew), dtype=float) if not is_categorical(Xnew) else None
    if X is None or X.ndim != 2 or X.shape[1] != len(md.preds):
        if X is not None and X.ndim == 2 and X.shape[0] == len(md.preds) and X.shape[1] == 1:
            X = X.T
        else:
            raise MatlabError(f"{fname}: the new data must have {len(md.preds)} column(s), one per predictor.")
    for c, j in enumerate(md.preds):
        vals[j] = _encode(md, j, X[:, c].reshape(-1, 1))
    return vals


def _encode(md: ModelData, j: int, v: Any) -> np.ndarray:
    if md.levels[j] is None:
        return np.asarray(fnum(v), dtype=float).flatten(order="F")
    if is_categorical(v) or text_list(v) is not None:
        names = list(v.names_f()) if is_categorical(v) else text_list(v)
    else:
        names = [num_text(x) for x in vec(v)]
    lv = md.levels[j]
    return np.array([lv.index(s) if s in lv else -1 for s in names], dtype=int)


# -- least squares core ---------------------------------------------------------------------------------------------

class LSFit:
    """Weighted least squares with the summary statistics shared by the linear-model functions."""

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, has_intercept: bool = True):
        n, p = X.shape
        self.X, self.y = X, y
        self.w = np.ones(n) if w is None else w
        sw = np.sqrt(self.w)
        Xw, yw = X * sw[:, None], y * sw
        if p:
            self.beta, _, self.rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            self.rank = int(self.rank)
            self.xtxi = np.linalg.pinv(Xw.T @ Xw)
        else:
            self.beta, self.rank, self.xtxi = np.zeros(0), 0, np.zeros((0, 0))
        self.fitted = X @ self.beta
        self.resid = y - self.fitted
        self.n = int(np.sum(self.w > 0))
        self.dfe = self.n - self.rank
        self.sse = float(np.sum(self.w * self.resid ** 2))
        ybar = float(np.sum(self.w * y) / np.sum(self.w)) if has_intercept and n else 0.0
        self.sst = float(np.sum(self.w * (y - ybar) ** 2))
        self.ssr = self.sst - self.sse
        self.mse = self.sse / self.dfe if self.dfe > 0 else np.nan
        self.cov = self.mse * self.xtxi
        with np.errstate(all="ignore"):
            self.se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
            self.t = self.beta / self.se
        self.lev = np.einsum("ij,jk,ik->i", Xw, self.xtxi, Xw) if p else np.zeros(n)
        self.has_intercept = has_intercept

    def pvalues(self) -> np.ndarray:
        import scipy.stats as st
        with np.errstate(all="ignore"):
            return 2 * st.t.sf(np.abs(self.t), self.dfe) if self.dfe > 0 else np.full(self.t.shape, np.nan)


def _coef_table(names: list[str], est, se, t, p, tname: str = "tStat"):
    return make_table(["Estimate", "SE", tname, "pValue"], [np.asarray(est, dtype=float), np.asarray(se, dtype=float),
                                                           np.asarray(t, dtype=float), np.asarray(p, dtype=float)],
                      row_names=list(names))


def _fmt_g(x: float, digits: int = 5) -> str:
    if np.isnan(x):
        return "NaN"
    return f"{x:.{digits}g}"


def _coef_lines(names, cols: dict) -> list[str]:
    heads = list(cols)
    body = [[_fmt_g(float(cols[h][i])) for h in heads] for i in range(len(names))]
    wn = max([len(n) for n in names] + [0])
    widths = [max([len(h)] + [len(r[j]) for r in body]) for j, h in enumerate(heads)]
    lines = ["    " + " " * wn + "    " + "    ".join(h.center(widths[j]) for j, h in enumerate(heads)),
             "    " + " " * wn + "    " + "    ".join("_" * widths[j] for j in range(len(heads))), ""]
    for n, r in zip(names, body):
        lines.append("    " + n.ljust(wn) + "    " + "    ".join(r[j].rjust(widths[j]) for j in range(len(heads))))
    return lines


# -- LinearModel ----------------------------------------------------------------------------------------------------

_LM_OPTS = ["CategoricalVars", "Exclude", "Intercept", "PredictorVars", "ResponseVar", "RobustOpts", "VarNames",
            "Weights"]


def _row_mask(v: Any, n: int, what: str) -> np.ndarray:
    a = np.asarray(v)
    mask = np.zeros(n, dtype=bool)
    if a.dtype == bool:
        if a.size != n:
            raise MatlabError(f"{what} must have one value per observation.")
        return a.flatten().copy()
    for i in vec(v):
        if not 1 <= int(i) <= n:
            raise MatlabError(f"{what} contains an index outside the data.")
        mask[int(i) - 1] = True
    return mask


class LinearModel(StatObject):
    matlab_class = "LinearModel"

    def __init__(self, md: ModelData, terms: list, intercept: bool, weights=None, exclude=None, robust=None):
        super().__init__()
        self.md, self.terms, self.intercept = md, list(terms), intercept
        self.weights = np.ones(md.n) if weights is None else weights
        self.exclude = np.zeros(md.n, dtype=bool) if exclude is None else exclude
        self.robust = robust
        self.steps = None
        self._fit()

    def _used_vars(self) -> list[int]:
        used = sorted({j for t in self.terms for j, k in enumerate(t) if k})
        return used + [self.md.resp]

    def _fit(self) -> None:
        md = self.md
        if md.levels[md.resp] is not None:
            raise MatlabError("The response of a linear model must be numeric.")
        self.missing = md.missing(self._used_vars()) | np.isnan(self.weights)
        self.subset = ~(self.missing | self.exclude)
        vals = [v[self.subset] if v is not None else None for v in md.values]
        X, self.cnames, self.owner = design(md, self.terms, self.intercept, vals)
        y = np.asarray(md.values[md.resp], dtype=float)[self.subset]
        w = self.weights[self.subset]
        if self.robust is not None:
            from .stats_regress import robust_irls
            rw = robust_irls(X, y, self.robust[0], self.robust[1], w)[1]
            self.robust_w = rw
            w = w * rw
        self.ls = LSFit(X, y, w, self.intercept)
        self.Xd = X

    # -- properties --------------------------------------------------------------------------------
    def property_names(self) -> list[str]:
        return ["Coefficients", "CoefficientNames", "CoefficientCovariance", "NumCoefficients",
                "NumEstimatedCoefficients", "NumObservations", "NumPredictors", "NumVariables", "DFE", "MSE", "RMSE",
                "SSE", "SSR", "SST", "Rsquared", "LogLikelihood", "ModelCriterion", "Fitted", "Residuals",
                "Diagnostics", "Formula", "PredictorNames", "ResponseName", "VariableNames", "Variables",
                "ObservationInfo", "Robust", "Steps"]

    def _full(self, v: np.ndarray) -> np.ndarray:
        out = np.full(self.md.n, np.nan)
        out[self.subset] = v
        return out.reshape(-1, 1)

    def _loglik(self) -> float:
        ls = self.ls
        n = ls.n
        if n == 0 or ls.sse <= 0:
            return np.inf if n else np.nan
        return float(-n / 2 * (np.log(2 * np.pi) + np.log(ls.sse / n) + 1) + 0.5 * np.sum(np.log(ls.w[ls.w > 0])))

    def _criteria(self, ll: float, k: int, n: int) -> Struct:
        with np.errstate(all="ignore"):
            aic = -2 * ll + 2 * k
            return struct(AIC=aic, AICc=aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan,
                          BIC=-2 * ll + k * np.log(n), CAIC=-2 * ll + k * (np.log(n) + 1))

    def _rsq(self) -> tuple[float, float]:
        ls = self.ls
        with np.errstate(all="ignore"):
            r2 = 1 - ls.sse / ls.sst if ls.sst > 0 else np.nan
            dft = ls.n - (1 if self.intercept else 0)
            adj = 1 - (ls.sse / ls.dfe) / (ls.sst / dft) if ls.dfe > 0 and ls.sst > 0 else np.nan
        return float(r2), float(adj)

    def _residual_parts(self):
        ls = self.ls
        with np.errstate(all="ignore"):
            raw = ls.resid
            rmse = np.sqrt(ls.mse)
            h = ls.lev
            std = raw / (rmse * np.sqrt(1 - h))
            s2i = (ls.sse - ls.w * raw ** 2 / (1 - h)) / (ls.dfe - 1) if ls.dfe > 1 else np.full(raw.shape, np.nan)
            stud = raw * np.sqrt(ls.w) / np.sqrt(s2i * (1 - h))
        return raw, raw * np.sqrt(ls.w) / rmse, std * np.sqrt(ls.w), stud, s2i

    def get_field(self, interp, name: str) -> Any:
        ls, md = self.ls, self.md
        if name == "Coefficients":
            return _coef_table(self.cnames, ls.beta, ls.se, ls.t, ls.pvalues())
        if name == "CoefficientNames":
            return cellstr(self.cnames, column=False)
        if name == "CoefficientCovariance":
            return ls.cov.copy()
        if name == "NumCoefficients":
            return float(len(self.cnames))
        if name == "NumEstimatedCoefficients":
            return float(ls.rank)
        if name == "NumObservations":
            return float(ls.n)
        if name == "NumPredictors":
            return float(len(md.preds))
        if name == "NumVariables":
            return float(len(md.names))
        if name in ("DFE", "MSE", "SSE", "SSR", "SST"):
            return float(getattr(ls, name.lower()))
        if name == "RMSE":
            return float(np.sqrt(ls.mse))
        if name == "Rsquared":
            r2, adj = self._rsq()
            return struct(Ordinary=r2, Adjusted=adj)
        if name == "LogLikelihood":
            return self._loglik()
        if name == "ModelCriterion":
            return self._criteria(self._loglik(), ls.rank, ls.n)
        if name == "Fitted":
            return self._full(ls.fitted)
        if name == "Residuals":
            raw, pear, std, stud, _ = self._residual_parts()
            return make_table(["Raw", "Pearson", "Studentized", "Standardized"],
                              [self._full(raw), self._full(pear), self._full(stud), self._full(std)])
        if name == "Diagnostics":
            raw, _, _, stud, s2i = self._residual_parts()
            h = ls.lev
            p = max(ls.rank, 1)
            with np.errstate(all="ignore"):
                cook = ls.w * raw ** 2 * h / (p * ls.mse * (1 - h) ** 2)
                dffits = stud * np.sqrt(h / (1 - h))
                covratio = (s2i / ls.mse) ** p / (1 - h)
            return make_table(["Leverage", "CooksDistance", "Dffits", "S2_i", "CovRatio"],
                              [self._full(h), self._full(cook), self._full(dffits), self._full(s2i),
                                      self._full(covratio)])
        if name == "Formula":
            return char(formula_text(md, self.terms, self.intercept))
        if name == "PredictorNames":
            return cellstr([md.names[j] for j in md.preds])
        if name == "ResponseName":
            return char(md.names[md.resp])
        if name == "VariableNames":
            return cellstr(md.names)
        if name == "Variables":
            return self._variables_table()
        if name == "ObservationInfo":
            return make_table(["Weights", "Excluded", "Missing", "Subset"],
                              [self.weights.reshape(-1, 1), self.exclude.reshape(-1, 1), self.missing.reshape(-1, 1),
                               self.subset.reshape(-1, 1)])
        if name == "Robust":
            if self.robust is None:
                return np.zeros((0, 0))
            return struct(RobustWgtFun=self.robust[0], Tune=self.robust[1], Weights=self._full(self.robust_w))
        if name == "Steps":
            return self.steps if self.steps is not None else np.zeros((0, 0))
        return super().get_field(interp, name)

    def _variables_table(self):
        from ._stats_util import make_categorical
        md = self.md
        cols = [make_categorical(v, md.levels[j]) if md.levels[j] is not None else np.asarray(v,
                dtype=float).reshape(-1, 1)
                for j, v in enumerate(md.values)]
        return make_table(md.names, cols)

    # -- methods ------------------------------------------------------------------------------------
    def _design_new(self, Xnew: Any, fname: str) -> np.ndarray:
        vals = new_values(self.md, Xnew, fname)
        X, _, _ = design(self.md, self.terms, self.intercept, vals)
        bad = np.zeros(X.shape[0], dtype=bool)
        for j in self.md.preds:
            if vals[j] is not None and self.md.levels[j] is not None:
                bad |= np.asarray(vals[j]) < 0
        X[bad] = np.nan
        return X

    def m_predict(self, interp, args, nargout):
        import scipy.stats as st
        pos, opts = get_opts(args, 0, ["Alpha", "Prediction", "Simultaneous"], "predict")
        if len(pos) > 1:
            raise MatlabError("Too many input arguments for predict.")
        X = self._design_new(pos[0], "predict") if pos else self.Xd
        ls = self.ls
        ypred = (X @ ls.beta).reshape(-1, 1)
        if not pos:
            ypred = self._full(ypred.flatten())
        if nargout < 2:
            return ypred
        alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
        obs = choice(opts["prediction"], ["curve", "observation"], "Prediction") == "observation" \
            if "prediction" in opts else False
        simul = as_bool(opts["simultaneous"], "Simultaneous") if "simultaneous" in opts else False
        var = np.einsum("ij,jk,ik->i", X, ls.cov, X) + (ls.mse if obs else 0.0)
        if simul:
            k = ls.rank + (1 if obs else 0)
            crit = np.sqrt(k * st.f.ppf(1 - alpha, k, ls.dfe))
        else:
            crit = st.t.ppf(1 - alpha / 2, ls.dfe)
        half = (crit * np.sqrt(var)).reshape(-1, 1)
        yv = (X @ ls.beta).reshape(-1, 1)
        return ypred if nargout < 2 else (yv, np.hstack([yv - half, yv + half]))

    def m_feval(self, interp, args, nargout):
        if not args:
            raise MatlabError("feval: the predictor values are missing.")
        if len(args) == 1:
            return (self._design_new(args[0], "feval") @ self.ls.beta).reshape(-1, 1)
        if len(args) != len(self.md.preds):
            raise MatlabError(f"feval: expected {len(self.md.preds)} predictor arguments.")
        shape = np.asarray(fnum(args[0])).shape
        X = np.column_stack([vec(a) for a in args])
        return (self._design_new(X, "feval") @ self.ls.beta).reshape(shape, order="F")

    def m_random(self, interp, args, nargout):
        if len(args) > 1:
            raise MatlabError("Too many input arguments for random.")
        X = self._design_new(args[0], "random") if args else self.Xd
        mu = X @ self.ls.beta
        return (mu + np.sqrt(self.ls.mse) * rng_of(interp).standard_normal(mu.size)).reshape(-1, 1)

    def m_coefCI(self, interp, args, nargout):
        import scipy.stats as st
        alpha = scalar(args[0], "alpha") if args else 0.05
        if len(args) > 1:
            raise MatlabError("Too many input arguments for coefCI.")
        ls = self.ls
        c = st.t.ppf(1 - alpha / 2, ls.dfe)
        return np.column_stack([ls.beta - c * ls.se, ls.beta + c * ls.se])

    def m_coefTest(self, interp, args, nargout):
        import scipy.stats as st
        ls = self.ls
        p = len(self.cnames)
        if len(args) > 2:
            raise MatlabError("Too many input arguments for coefTest.")
        if args:
            H = np.atleast_2d(np.asarray(fnum(args[0]), dtype=float))
            if H.shape[1] != p:
                raise MatlabError(f"coefTest: H must have {p} columns.")
        else:
            H = np.eye(p)[[i for i, o in enumerate(self.owner) if o >= 0]]
        c = vec(args[1]) if len(args) > 1 else np.zeros(H.shape[0])
        d = H @ ls.beta - c
        V = H @ ls.cov @ H.T
        r = int(np.linalg.matrix_rank(V))
        with np.errstate(all="ignore"):
            F = float(d @ np.linalg.pinv(V) @ d / r) if r else np.nan
        pv = float(st.f.sf(F, r, ls.dfe)) if r else np.nan
        return outs(nargout, pv, F, float(r))

    def _term_ss(self, sstype) -> list[tuple[float, int]]:
        """Sum of squares and degrees of freedom of every term."""
        ls = self.ls
        y, w = ls.y, ls.w
        sw = np.sqrt(w)

        def rss(cols: list[int]) -> tuple[float, int]:
            if not cols:
                return float(np.sum(w * y ** 2)), 0
            A = self.Xd[:, cols] * sw[:, None]
            coef, _, rank, _ = np.linalg.lstsq(A, y * sw, rcond=None)
            r = y * sw - A @ coef
            return float(r @ r), int(rank)

        owner = np.array(self.owner)
        out = []
        for i, t in enumerate(self.terms):
            if sstype == 1:
                small = [c for c, o in enumerate(owner) if o < i]
                big = [c for c, o in enumerate(owner) if o <= i]
            elif sstype == 3:
                big = list(range(owner.size))
                small = [c for c, o in enumerate(owner) if o != i]
            else:  # hierarchical: leave out the terms that contain this one
                above = {k for k, u in enumerate(self.terms) if k != i and all(a >= b for a, b in zip(u, t))}
                big = [c for c, o in enumerate(owner) if o not in above]
                small = [c for c in big if owner[c] != i]
            (r0, k0), (r1, k1) = rss(small), rss(big)
            out.append((max(r0 - r1, 0.0), k1 - k0))
        return out

    def m_anova(self, interp, args, nargout):
        import scipy.stats as st
        ls = self.ls
        kind = "component"
        sstype: Any = "h"
        rest = list(args)
        if rest and is_text(rest[0]) and text(rest[0]).lower() in ("component", "summary"):
            kind = text(rest.pop(0)).lower()
        if rest:
            v = rest.pop(0)
            sstype = text(v).lower() if is_text(v) else intval(v, "sstype")
            if sstype not in (1, 2, 3, "h"):
                raise MatlabError("anova: the sum-of-squares type must be 1, 2, 3 or 'h'.")
        if rest:
            raise MatlabError("Too many input arguments for anova.")
        with np.errstate(all="ignore"):
            if kind == "summary":
                dfm = ls.rank - (1 if self.intercept else 0)
                F = (ls.ssr / dfm) / ls.mse if dfm else np.nan
                rows = ["Total", "Model", "Residual"]
                ss = [ls.sst, ls.ssr, ls.sse]
                df = [ls.n - (1 if self.intercept else 0), dfm, ls.dfe]
                ms = [ss[0] / df[0] if df[0] else np.nan, ls.ssr / dfm if dfm else np.nan, ls.mse]
                Fs = [np.nan, F, np.nan]
                ps = [np.nan, float(st.f.sf(F, dfm, ls.dfe)) if dfm else np.nan, np.nan]
            else:
                parts = self._term_ss(2 if sstype in (2, "h") else sstype)
                rows = [term_name(t, self.md.names) for t in self.terms] + ["Error"]
                ss = [p[0] for p in parts] + [ls.sse]
                df = [p[1] for p in parts] + [ls.dfe]
                ms = [s / d if d else np.nan for s, d in zip(ss, df)]
                Fs = [m / ls.mse for m in ms[:-1]] + [np.nan]
                ps = [float(st.f.sf(f, d, ls.dfe)) if d else np.nan for f, d in zip(Fs[:-1], df[:-1])] + [np.nan]
        return make_table(["SumSq", "DF", "MeanSq", "F", "pValue"],
                          [np.array(ss, dtype=float), np.array(df, dtype=float), np.array(ms, dtype=float),
                           np.array(Fs, dtype=float), np.array(ps, dtype=float)], row_names=rows)

    def m_dwtest(self, interp, args, nargout):
        import scipy.stats as st
        if args:
            raise MatlabError("dwtest: options are not supported for model objects.")
        r = self.ls.resid
        dw = float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))
        n, k = r.size, self.ls.rank
        # normal approximation to the null distribution of the statistic
        z = (dw - 2) / np.sqrt(4.0 * (n - 2) / ((n - k) * (n - k + 2)) if n - k > 0 else np.nan)
        return outs(nargout, float(min(1.0, 2 * st.norm.sf(abs(z)))), dw)

    def _refit(self, terms: list):
        m = LinearModel(self.md, _sort_terms(terms), self.intercept, self.weights, self.exclude, self.robust)
        return m

    def _terms_arg(self, v: Any, fname: str) -> list:
        if is_text(v):
            p = _FormulaParser(_tokens(text(v)), self.md.names)
            terms = p.expr()
            return [t for t in terms if any(t)]
        T = np.atleast_2d(np.asarray(fnum(v))).astype(int)
        if T.shape[1] != len(self.md.names):
            raise MatlabError(f"{fname}: a terms matrix must have {len(self.md.names)} columns.")
        return [tuple(int(x) for x in r) for r in T if np.any(r)]

    def m_addTerms(self, interp, args, nargout):
        if len(args) != 1:
            raise MatlabError("addTerms: expected addTerms(mdl, terms).")
        new = self._terms_arg(args[0], "addTerms")
        return self._refit(self.terms + [t for t in new if t not in self.terms])

    def m_removeTerms(self, interp, args, nargout):
        if len(args) != 1:
            raise MatlabError("removeTerms: expected removeTerms(mdl, terms).")
        drop = self._terms_arg(args[0], "removeTerms")
        return self._refit([t for t in self.terms if t not in drop])

    def m_plot(self, interp, args, nargout):
        if args:
            raise MatlabError("plot: options are not supported for model objects.")
        md, ls = self.md, self.ls
        y = ls.y
        numeric_single = len(self.md.preds) == 1 and md.levels[md.preds[0]] is None
        if numeric_single:
            x = np.asarray(md.values[md.preds[0]], dtype=float)[self.subset]
            xs = np.linspace(x.min(), x.max(), 100)
            xlabel = md.names[md.preds[0]]
            Xg = self._design_new(xs.reshape(-1, 1), "plot")
        else:
            x = ls.fitted
            xs = np.linspace(x.min(), x.max(), 100) if x.size else np.zeros(0)
            xlabel = "Fitted values"
            Xg = None
        h = builtin_call(interp, "plot", [x.reshape(1, -1), y.reshape(1, -1), char("x")], 1)
        builtin_call(interp, "hold", [char("on")], 0)
        if Xg is not None:
            import scipy.stats as st
            fit = Xg @ ls.beta
            half = st.t.ppf(0.975, ls.dfe) * np.sqrt(np.einsum("ij,jk,ik->i", Xg, ls.cov,
                    Xg)) if ls.dfe > 0 else 0 * fit
            builtin_call(interp, "plot", [xs.reshape(1, -1), fit.reshape(1, -1), char("r-")], 0)
            builtin_call(interp, "plot", [xs.reshape(1, -1), (fit - half).reshape(1, -1), char("r:")], 0)
            builtin_call(interp, "plot", [xs.reshape(1, -1), (fit + half).reshape(1, -1), char("r:")], 0)
        else:
            builtin_call(interp, "plot", [xs.reshape(1, -1), xs.reshape(1, -1), char("r-")], 0)
        builtin_call(interp, "hold", [char("off")], 0)
        builtin_call(interp, "xlabel", [char(xlabel)], 0)
        builtin_call(interp, "ylabel", [char(md.names[md.resp])], 0)
        builtin_call(interp, "title", [char(f"{md.names[md.resp]} vs. {xlabel}")], 0)
        return h[0] if nargout else None

    def m_plotResiduals(self, interp, args, nargout):
        pos, opts = get_opts(args, 0, ["ResidualType"], "plotResiduals")
        kinds = ["histogram", "fitted", "caseorder", "probability", "lagged", "symmetry", "observed"]
        kind = choice(pos[0], kinds, "The plot type") if pos else "histogram"
        raw, pear, std, stud, _ = self._residual_parts()
        rtype = choice(opts["residualtype"], ["Raw", "Pearson", "Standardized", "Studentized"], "ResidualType") \
            if "residualtype" in opts else "raw"
        r = {"raw": raw, "pearson": pear, "standardized": std, "studentized": stud}[rtype]
        r = r[~np.isnan(r)]
        if kind == "histogram":
            h = builtin_call(interp, "histogram", [r.reshape(-1, 1)], 1)
            builtin_call(interp, "xlabel", [char("Residuals")], 0)
        elif kind == "probability":
            h = builtin_call(interp, "normplot", [r.reshape(-1, 1)], 1)
        elif kind == "symmetry":
            med = np.median(r)
            lo, hi = np.sort(med - r[r < med]), np.sort(r[r > med] - med)
            k = min(lo.size, hi.size)
            h = builtin_call(interp, "plot", [lo[:k].reshape(1, -1), hi[:k].reshape(1, -1), char("x")], 1)
        else:
            if kind == "fitted":
                x, lab = self.ls.fitted, "Fitted values"
            elif kind == "observed":
                x, lab = self.ls.y, "Observed values"
            elif kind == "lagged":
                x, r, lab = r[:-1], r[1:], "Residual(t-1)"
            else:
                x, lab = np.arange(1, r.size + 1.0), "Row number"
            h = builtin_call(interp, "plot", [np.asarray(x[:r.size]).reshape(1, -1), r.reshape(1, -1), char("x")], 1)
            builtin_call(interp, "xlabel", [char(lab)], 0)
            builtin_call(interp, "ylabel", [char("Residuals")], 0)
        return h[0] if nargout else None

    def m_plotDiagnostics(self, interp, args, nargout):
        kind = choice(args[0], ["leverage", "cookd", "dffits", "s2_i", "covratio"], "The plot type") if args \
            else "leverage"
        if len(args) > 1:
            raise MatlabError("Too many input arguments for plotDiagnostics.")
        tbl = self.get_field(interp, "Diagnostics")
        col = {"leverage": "Leverage", "cookd": "CooksDistance", "dffits": "Dffits", "s2_i": "S2_i",
               "covratio": "CovRatio"}[kind]
        v = np.asarray(tbl.columns[tbl.names.index(col)], dtype=float).flatten()
        h = builtin_call(interp, "plot", [np.arange(1, v.size + 1.0).reshape(1, -1), v.reshape(1, -1), char("x")], 1)
        builtin_call(interp, "xlabel", [char("Row number")], 0)
        builtin_call(interp, "ylabel", [char(col)], 0)
        return h[0] if nargout else None

    def display_lines(self) -> list[str]:
        import scipy.stats as st
        ls = self.ls
        lines = ["", "Linear regression model" + (" (robust fit):" if self.robust else ":"),
                 "    " + formula_text(self.md, self.terms, self.intercept), "", "Estimated Coefficients:"]
        lines += _coef_lines(self.cnames, {"Estimate": ls.beta, "SE": ls.se, "tStat": ls.t, "pValue": ls.pvalues()})
        r2, adj = self._rsq()
        lines += ["", "", f"Number of observations: {ls.n}, Error degrees of freedom: {ls.dfe}",
                  f"Root Mean Squared Error: {_fmt_g(float(np.sqrt(ls.mse)), 3)}"]
        if self.intercept:
            lines.append(f"R-squared: {_fmt_g(r2, 3)},  Adjusted R-Squared: {_fmt_g(adj, 3)}")
            dfm = ls.rank - 1
            if dfm > 0 and ls.dfe > 0:
                with np.errstate(all="ignore"):
                    F = (ls.ssr / dfm) / ls.mse
                lines.append(f"F-statistic vs. constant model: {_fmt_g(float(F), 3)}, "
                             f"p-value = {_fmt_g(float(st.f.sf(F, dfm, ls.dfe)), 3)}")
        return lines

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


def _lm_common(pos: list, opts: dict, fname: str):
    md, rest = model_data(pos, opts, fname)
    spec = rest[0] if rest else None
    if len(rest) > 1:
        raise MatlabError(f"Too many input arguments for {fname}.")
    terms, intercept = spec_terms(spec, md)
    if "intercept" in opts:
        intercept = intercept and as_bool(opts["intercept"], "Intercept")
    weights = None
    if "weights" in opts:
        weights = vec(opts["weights"])
        if weights.size != md.n or np.any(weights < 0):
            raise MatlabError(f"{fname}: Weights must hold one non-negative value per observation.")
    exclude = _row_mask(opts["exclude"], md.n, "Exclude") if "exclude" in opts else None
    return md, terms, intercept, weights, exclude


def _robust_opt(opts: dict):
    if "robustopts" not in opts:
        return None
    v = opts["robustopts"]
    if is_text(v):
        s = text(v).lower()
        if s == "off":
            return None
        if s == "on":
            return ("bisquare", None)
        return (s, None)
    if isinstance(v, Struct):
        fn = text(v.get_field(0, "RobustWgtFun")) if "RobustWgtFun" in v.fields else "bisquare"
        tune = scalar(v.get_field(0, "Tune"), "Tune") if "Tune" in v.fields and not is_empty_arg(
            v.get_field(0, "Tune")) else None
        return (fn, tune)
    raise MatlabError("RobustOpts must be 'on', 'off', a weight function name or a structure.")


@builtin("fitlm")
def _fitlm(interp, args, nargout):
    pos, opts = get_opts(args, 1, _LM_OPTS, "fitlm")
    md, terms, intercept, weights, exclude = _lm_common(pos, opts, "fitlm")
    return LinearModel(md, terms, intercept, weights, exclude, _robust_opt(opts))


def _term_pvalue(big: LinearModel, small_sse: float, small_rank: int) -> tuple[float, float]:
    import scipy.stats as st
    ls = big.ls
    d = ls.rank - small_rank
    if d <= 0 or ls.dfe <= 0:
        return 1.0, 0.0
    with np.errstate(all="ignore"):
        F = ((small_sse - ls.sse) / d) / ls.mse
    return float(st.f.sf(F, d, ls.dfe)) if np.isfinite(F) else 0.0, float(F)


@builtin("stepwiselm")
def _stepwiselm(interp, args, nargout):
    known = _LM_OPTS + ["Criterion", "Lower", "Upper", "NSteps", "PEnter", "PRemove", "Verbose"]
    pos, opts = get_opts(args, 1, known, "stepwiselm")
    crit = choice(opts["criterion"], ["sse", "aic", "bic", "rsquared", "adjrsquared"], "Criterion") \
        if "criterion" in opts else "sse"
    if crit != "sse":
        raise MatlabError("stepwiselm: only 'Criterion', 'sse' (F-test p-values) is supported.")
    penter = scalar(opts["penter"], "PEnter") if "penter" in opts else 0.05
    premove = scalar(opts["premove"], "PRemove") if "premove" in opts else 0.10
    if penter >= premove:
        raise MatlabError("stepwiselm: PEnter must be smaller than PRemove.")
    nsteps = intval(opts["nsteps"], "NSteps") if "nsteps" in opts else 10 ** 6
    verbose = intval(opts["verbose"], "Verbose") if "verbose" in opts else 1
    md, rest = model_data(pos, opts, "stepwiselm")
    start_spec = rest[0] if rest else char("constant")
    if len(rest) > 1:
        raise MatlabError("Too many input arguments for stepwiselm.")
    terms, intercept = spec_terms(start_spec, md)
    upper, _ = spec_terms(opts.get("upper", char("interactions")), md)
    lower, _ = spec_terms(opts.get("lower", char("constant")), md)
    upper = upper + [t for t in terms if t not in upper]
    weights = vec(opts["weights"]) if "weights" in opts else None
    exclude = _row_mask(opts["exclude"], md.n, "Exclude") if "exclude" in opts else None
    # use the same rows for every candidate model
    used = sorted({j for t in upper for j, k in enumerate(t) if k} | {md.resp})
    miss = md.missing(used)
    exclude = miss if exclude is None else (exclude | miss)

    def fit(ts):
        return LinearModel(md, _sort_terms(ts), intercept, weights, exclude)

    def allowed_add(ts, t):
        # a term may enter once all the terms it is built from are in the model
        subs = [u for u in upper if u != t and all(a <= b for a, b in zip(u, t)) and any(u)]
        return all(u in ts for u in subs)

    def allowed_drop(ts, t):
        return t not in lower and not any(u != t and all(a >= b for a, b in zip(u, t)) for u in ts)

    cur = fit(terms)
    history = []
    for _ in range(nsteps):
        best = None
        for t in upper:
            if t in cur.terms or not allowed_add(cur.terms, t):
                continue
            cand = fit(cur.terms + [t])
            pv, F = _term_pvalue(cand, cur.ls.sse, cur.ls.rank)
            if pv < penter and (best is None or pv < best[0]):
                best = (pv, F, t, cand, "Adding")
        if best is None:
            worst = None
            for t in cur.terms:
                if not allowed_drop(cur.terms, t):
                    continue
                small = fit([u for u in cur.terms if u != t])
                pv, F = _term_pvalue(cur, small.ls.sse, small.ls.rank)
                if pv > premove and (worst is None or pv > worst[0]):
                    worst = (pv, F, t, small, "Removing")
            best = worst
        if best is None:
            break
        pv, F, t, cur, action = best
        history.append((action, term_name(t, md.names), F, pv))
        if verbose:
            interp.out(f"{len(history)}. {action} {term_name(t, md.names)}, FStat = {_fmt_g(F)}, "
                       f"pValue = {_fmt_g(pv)}\n")
    cur.steps = struct(Start=formula_text(md, _sort_terms(terms), intercept),
                       Lower=formula_text(md, lower, intercept), Upper=formula_text(md, upper, intercept),
                       Criterion="SSE", PEnter=penter, PRemove=premove,
                       History=make_table(["Action", "TermName", "FStat", "pValue"],
                                          [[h[0] for h in history], [h[1] for h in history],
                                           np.array([h[2] for h in history], dtype=float),
                                           np.array([h[3] for h in history], dtype=float)]) if history
                       else np.zeros((0, 0)))
    return cur


@builtin("x2fx")
def _x2fx(interp, args, nargout):
    if not 1 <= len(args) <= 4:
        raise MatlabError("x2fx: expected x2fx(X), x2fx(X, model) or x2fx(X, model, categ, catlevels).")
    X = np.asarray(fnum(args[0]), dtype=float)
    if len(args) > 2 and not is_empty_arg(args[2]):
        raise MatlabError("x2fx: categorical predictors are not supported; use fitlm with 'CategoricalVars'.")
    md = ModelData()
    for j in range(X.shape[1]):
        md.add(f"x{j + 1}", X[:, j].reshape(-1, 1))
    md.names.append("y")
    md.values.append(np.zeros(X.shape[0]))
    md.levels.append(None)
    md.resp = X.shape[1]
    md.preds = list(range(X.shape[1]))
    spec = args[1] if len(args) > 1 else char("linear")
    if is_text(spec) and text(spec).lower() == "interaction":
        spec = char("interactions")
    terms, icpt = spec_terms(spec, md)
    D, _, _ = design(md, terms, icpt)
    return D


# -- generalized linear models -------------------------------------------------------------------------------

_DISTS = ["normal", "binomial", "poisson", "gamma", "inverse gaussian"]
_CANON = {"normal": "identity", "binomial": "logit", "poisson": "log", "gamma": "reciprocal",
          "inverse gaussian": -2.0}


class Link:
    def __init__(self, spec: Any, dist: str):
        if spec is None or is_empty_arg(spec):
            spec = _CANON[dist]
        if isinstance(spec, str) or is_text(spec):
            s = spec if isinstance(spec, str) else text(spec).lower()
            s = {"comploglog": "comploglog", "cloglog": "comploglog"}.get(s, s)
            if s not in ("identity", "log", "logit", "probit", "comploglog", "loglog", "reciprocal"):
                raise MatlabError(f"Unknown link function '{s}'.")
            self.name: Any = s
        elif isinstance(spec, Cell) or isinstance(spec, Struct):
            raise MatlabError("Custom link functions are not supported.")
        else:
            self.name = float(scalar(spec, "the link exponent"))
            if self.name == 1:
                self.name = "identity"
            elif self.name == 0:
                self.name = "log"
            elif self.name == -1:
                self.name = "reciprocal"

    def label(self) -> str:
        return self.name if isinstance(self.name, str) else f"power({self.name:g})"

    def g(self, mu):
        import scipy.special as sp
        n = self.name
        with np.errstate(all="ignore"):
            if n == "identity":
                return mu
            if n == "log":
                return np.log(mu)
            if n == "logit":
                return np.log(mu / (1 - mu))
            if n == "probit":
                return sp.ndtri(mu)
            if n == "comploglog":
                return np.log(-np.log1p(-mu))
            if n == "loglog":
                return np.log(-np.log(mu))
            if n == "reciprocal":
                return 1 / mu
            return mu ** n

    def inv(self, eta):
        import scipy.special as sp
        n = self.name
        with np.errstate(all="ignore"):
            if n == "identity":
                return eta
            if n == "log":
                return np.exp(eta)
            if n == "logit":
                return sp.expit(eta)
            if n == "probit":
                return sp.ndtr(eta)
            if n == "comploglog":
                return -np.expm1(-np.exp(eta))
            if n == "loglog":
                return np.exp(-np.exp(eta))
            if n == "reciprocal":
                return 1 / eta
            return eta ** (1 / n)

    def deriv(self, mu):
        """d eta / d mu."""
        import scipy.special as sp
        n = self.name
        with np.errstate(all="ignore"):
            if n == "identity":
                return np.ones_like(mu)
            if n == "log":
                return 1 / mu
            if n == "logit":
                return 1 / (mu * (1 - mu))
            if n == "probit":
                return 1 / np.exp(-0.5 * sp.ndtri(mu) ** 2) * np.sqrt(2 * np.pi)
            if n == "comploglog":
                return 1 / (-(1 - mu) * np.log1p(-mu))
            if n == "loglog":
                return 1 / (mu * np.log(mu))
            if n == "reciprocal":
                return -1 / mu ** 2
            return n * mu ** (n - 1)


def _variance(dist: str, mu):
    if dist == "normal":
        return np.ones_like(mu)
    if dist == "binomial":
        return mu * (1 - mu)
    if dist == "poisson":
        return mu
    if dist == "gamma":
        return mu ** 2
    return mu ** 3


def _deviance_terms(dist: str, y, mu, N):
    with np.errstate(all="ignore"):
        if dist == "normal":
            return (y - mu) ** 2
        if dist == "binomial":
            a = np.where(y > 0, y * np.log(y / mu), 0.0)
            b = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
            return 2 * N * (a + b)
        if dist == "poisson":
            return 2 * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
        if dist == "gamma":
            return 2 * (-np.log(y / mu) + (y - mu) / mu)
        return (y - mu) ** 2 / (mu ** 2 * y)


class GLMFit:
    """Iteratively reweighted least squares for a generalized linear model."""

    def __init__(self, X, y, dist: str, link: Link, N=None, pw=None, offset=None, estdisp: bool | None = None,
                 max_iter: int = 100, tol: float = 1e-10):
        n, p = X.shape
        self.X, self.y, self.dist, self.link = X, y, dist, link
        self.N = np.ones(n) if N is None else N
        self.pw = np.ones(n) if pw is None else pw
        self.offset = np.zeros(n) if offset is None else offset
        if dist == "binomial":
            if np.any((y < 0) | (y > 1)):
                raise MatlabError("For the binomial distribution the response must lie between 0 and 1 "
                                  "(or be given as counts with the number of trials).")
            mu = (self.N * y + 0.5) / (self.N + 1)
        elif dist == "poisson":
            if np.any(y < 0):
                raise MatlabError("For the Poisson distribution the response must not be negative.")
            mu = y + 0.25
        elif dist in ("gamma", "inverse gaussian"):
            if np.any(y <= 0):
                raise MatlabError(f"For the {dist} distribution the response must be positive.")
            mu = np.maximum(y, 1e-10)
        else:
            mu = y.copy()
        eta = link.g(mu)
        if not np.all(np.isfinite(eta)):
            mu = np.full(n, max(float(np.mean(y)), 1e-3)) if dist != "binomial" else np.full(n, 0.5)
            eta = link.g(mu)
        beta = np.zeros(p)
        wprior = self.pw * (self.N if dist == "binomial" else 1.0)
        self.iterations = 0
        self.converged = False
        for it in range(max_iter):
            with np.errstate(all="ignore"):
                d = link.deriv(mu)
                z = eta - self.offset + (y - mu) * d
                W = wprior / (d ** 2 * _variance(dist, mu))
            W = np.where(np.isfinite(W), np.maximum(W, 0), 0.0)
            z = np.where(np.isfinite(z), z, 0.0)
            sw = np.sqrt(W)
            new, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None) if p else (np.zeros(0),)
            eta = X @ new + self.offset
            mu = link.inv(eta)
            if dist == "binomial":
                mu = np.clip(mu, 1e-15, 1 - 1e-15)
            elif dist != "normal":
                mu = np.maximum(mu, 1e-300)
            self.iterations = it + 1
            if np.all(np.abs(new - beta) <= tol * np.maximum(np.abs(beta), 1e-6)) and it > 0:
                beta = new
                self.converged = True
                break
            beta = new
            if dist == "normal" and link.name == "identity":
                self.converged = True
                break
        self.beta, self.mu, self.eta = beta, mu, eta
        with np.errstate(all="ignore"):
            d = link.deriv(mu)
            self.W = np.where(np.isfinite(wprior / (d ** 2 * _variance(dist, mu))),
                              wprior / (d ** 2 * _variance(dist, mu)), 0.0)
        sw = np.sqrt(self.W)
        Xw = X * sw[:, None]
        self.rank = int(np.linalg.matrix_rank(Xw)) if p else 0
        self.xtxi = np.linalg.pinv(Xw.T @ Xw) if p else np.zeros((0, 0))
        self.n = int(np.sum(self.pw > 0))
        self.dfe = max(self.n - self.rank, 0)
        self.dev = float(np.sum(self.pw * _deviance_terms(dist, y, mu, self.N)))
        with np.errstate(all="ignore"):
            self.pearson = (y - mu) * np.sqrt(wprior) / np.sqrt(_variance(dist, mu))
        self.sfit = float(np.sqrt(np.sum(self.pearson ** 2) / self.dfe)) if self.dfe > 0 else np.nan
        self.estdisp = dist in ("normal", "gamma", "inverse gaussian") if estdisp is None else estdisp
        self.s = self.sfit if self.estdisp else 1.0
        self.cov = self.xtxi * self.s ** 2
        with np.errstate(all="ignore"):
            self.se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
            self.t = beta / self.se

    def pvalues(self):
        import scipy.stats as st
        with np.errstate(all="ignore"):
            if self.estdisp:
                return 2 * st.t.sf(np.abs(self.t), self.dfe) if self.dfe > 0 else np.full(self.t.shape, np.nan)
            return 2 * st.norm.sf(np.abs(self.t))

    def loglik(self) -> float:
        from scipy.special import gammaln
        y, mu, N, w = self.y, self.mu, self.N, self.pw
        with np.errstate(all="ignore"):
            if self.dist == "normal":
                n = self.n
                sse = float(np.sum(w * (y - mu) ** 2))
                return float(-n / 2 * (np.log(2 * np.pi) + np.log(sse / n) + 1)) if sse > 0 else np.inf
            if self.dist == "binomial":
                k = N * y
                ll = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) + k * np.log(mu) + (N - k) * np.log1p(-mu)
            elif self.dist == "poisson":
                ll = y * np.log(mu) - mu - gammaln(y + 1)
            elif self.dist == "gamma":
                a = 1 / self.sfit ** 2
                ll = (a - 1) * np.log(y) - y * a / mu - a * np.log(mu / a) - gammaln(a)
            else:
                lam = 1 / self.sfit ** 2
                ll = 0.5 * np.log(lam / (2 * np.pi * y ** 3)) - lam * (y - mu) ** 2 / (2 * mu ** 2 * y)
            return float(np.sum(w * ll))


def _dist_name(v: Any) -> str:
    s = text(v).lower() if not isinstance(v, str) else v.lower()
    s = {"inversegaussian": "inverse gaussian", "inverse_gaussian": "inverse gaussian", "gaussian": "normal",
         "norm": "normal"}.get(s, s)
    if s not in _DISTS:
        raise MatlabError("The distribution must be 'normal', 'binomial', 'poisson', 'gamma' or 'inverse gaussian'.")
    return s


def _binomial_response(y: np.ndarray, N: np.ndarray | None, fname: str):
    """Proportions and trial counts from a 0/1 response, [successes trials] or counts with a size."""
    if N is None:
        return y, np.ones(y.size)
    with np.errstate(all="ignore"):
        prop = np.where(N > 0, y / np.where(N > 0, N, 1), 0.0)
    if np.any(y > N):
        raise MatlabError(f"{fname}: the number of successes exceeds the number of trials.")
    return prop, N


class GeneralizedLinearModel(StatObject):
    matlab_class = "GeneralizedLinearModel"

    def __init__(self, md: ModelData, terms: list, intercept: bool, dist: str, link: Link, weights, exclude,
                 offset, bsize, estdisp):
        super().__init__()
        self.md, self.terms, self.intercept = md, list(terms), intercept
        self.dist, self.link = dist, link
        n = md.n
        self.weights = np.ones(n) if weights is None else weights
        self.exclude = np.zeros(n, dtype=bool) if exclude is None else exclude
        self.offset = np.zeros(n) if offset is None else offset
        self.bsize = bsize
        used = sorted({j for t in terms for j, k in enumerate(t) if k}) + [md.resp]
        self.missing = md.missing(used) | np.isnan(self.weights) | np.isnan(self.offset)
        self.subset = ~(self.missing | self.exclude)
        vals = [v[self.subset] if v is not None else None for v in md.values]
        X, self.cnames, self.owner = design(md, terms, intercept, vals)
        yraw = np.asarray(md.values[md.resp], dtype=float)[self.subset] if md.levels[md.resp] is None \
            else np.asarray(md.values[md.resp], dtype=float)[self.subset]
        N = None
        if dist == "binomial":
            if md.levels[md.resp] is not None and len(md.levels[md.resp]) != 2:
                raise MatlabError("A categorical binomial response must have exactly two categories.")
            trials = bsize[self.subset] if bsize is not None else None
            y, N = _binomial_response(yraw, trials, "fitglm")
        else:
            y = yraw
        self.Xd = X
        self.fit = GLMFit(X, y, dist, link, N, self.weights[self.subset], self.offset[self.subset], estdisp)
        ones = np.ones((X.shape[0], 1)) if intercept else np.zeros((X.shape[0], 0))
        self.null = GLMFit(ones, y, dist, link, N, self.weights[self.subset], self.offset[self.subset], estdisp)

    def property_names(self) -> list[str]:
        return ["Coefficients", "CoefficientNames", "CoefficientCovariance", "NumCoefficients",
                "NumEstimatedCoefficients", "NumObservations", "NumPredictors", "NumVariables", "DFE", "Deviance",
                "Dispersion", "DispersionEstimated", "Distribution", "Link", "Offset", "Fitted", "Residuals",
                "LogLikelihood", "ModelCriterion", "Rsquared", "SSE", "SSR", "SST", "Formula", "PredictorNames",
                "ResponseName", "VariableNames", "Variables", "ObservationInfo"]

    def _full(self, v) -> np.ndarray:
        out = np.full(self.md.n, np.nan)
        out[self.subset] = v
        return out.reshape(-1, 1)

    def _ss(self):
        f = self.fit
        w = f.pw
        y = f.y * f.N
        mu = f.mu * f.N
        ybar = np.sum(w * y) / np.sum(w)
        sse, sst = float(np.sum(w * (y - mu) ** 2)), float(np.sum(w * (y - ybar) ** 2))
        return sse, sst - sse, sst

    def get_field(self, interp, name: str) -> Any:
        f, md = self.fit, self.md
        if name == "Coefficients":
            return _coef_table(self.cnames, f.beta, f.se, f.t, f.pvalues())
        if name == "CoefficientNames":
            return cellstr(self.cnames, column=False)
        if name == "CoefficientCovariance":
            return f.cov.copy()
        if name == "NumCoefficients":
            return float(len(self.cnames))
        if name == "NumEstimatedCoefficients":
            return float(f.rank)
        if name == "NumObservations":
            return float(f.n)
        if name == "NumPredictors":
            return float(len(md.preds))
        if name == "NumVariables":
            return float(len(md.names))
        if name == "DFE":
            return float(f.dfe)
        if name == "Deviance":
            return f.dev
        if name == "Dispersion":
            return float(f.s ** 2) if f.estdisp else 1.0
        if name == "DispersionEstimated":
            return bool(f.estdisp)
        if name == "Distribution":
            return struct(Name=self.dist)
        if name == "Link":
            return struct(Name=self.link.label())
        if name == "Offset":
            return self.offset.reshape(-1, 1)
        if name == "Fitted":
            cols = [self._full(f.mu * f.N), self._full(f.eta)]
            names = ["Response", "LinearPredictor"]
            if self.dist == "binomial":
                cols.append(self._full(f.mu))
                names.append("Probability")
            return make_table(names, cols)
        if name == "Residuals":
            with np.errstate(all="ignore"):
                raw = (f.y - f.mu) * f.N
                lin = (f.y - f.mu) * f.link.deriv(f.mu)
                dev = np.sign(f.y - f.mu) * np.sqrt(np.clip(_deviance_terms(self.dist, f.y, f.mu, f.N), 0, None))
            return make_table(["Raw", "LinearPredictor", "Pearson", "Deviance"],
                              [self._full(raw), self._full(lin), self._full(f.pearson), self._full(dev)])
        if name == "LogLikelihood":
            return f.loglik()
        if name == "ModelCriterion":
            ll, k, n = f.loglik(), f.rank, f.n
            with np.errstate(all="ignore"):
                aic = -2 * ll + 2 * k
                return struct(AIC=aic, AICc=aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan,
                              BIC=-2 * ll + k * np.log(n), CAIC=-2 * ll + k * (np.log(n) + 1))
        if name == "Rsquared":
            sse, _, sst = self._ss()
            ll, ll0 = f.loglik(), self.null.loglik()
            n = f.n
            with np.errstate(all="ignore"):
                r2 = 1 - sse / sst if sst > 0 else np.nan
                adj = 1 - (sse / f.dfe) / (sst / (n - 1)) if f.dfe > 0 and sst > 0 else np.nan
                gen = (1 - np.exp(2 * (ll0 - ll) / n)) / (1 - np.exp(2 * ll0 / n))
            return struct(Ordinary=r2, Adjusted=adj, LLR=1 - ll / ll0 if ll0 else np.nan,
                          Deviance=1 - f.dev / self.null.dev if self.null.dev else np.nan, AdjGeneralized=gen)
        if name in ("SSE", "SSR", "SST"):
            return self._ss()[("SSE", "SSR", "SST").index(name)]
        if name == "Formula":
            return char(formula_text(md, self.terms, self.intercept, self.link.label()))
        if name == "PredictorNames":
            return cellstr([md.names[j] for j in md.preds])
        if name == "ResponseName":
            return char(md.names[md.resp])
        if name == "VariableNames":
            return cellstr(md.names)
        if name == "Variables":
            return LinearModel._variables_table(self)
        if name == "ObservationInfo":
            return make_table(["Weights", "Excluded", "Missing", "Subset"],
                              [self.weights.reshape(-1, 1), self.exclude.reshape(-1, 1), self.missing.reshape(-1, 1),
                               self.subset.reshape(-1, 1)])
        return super().get_field(interp, name)

    _design_new = LinearModel._design_new

    def _crit(self, alpha: float) -> float:
        import scipy.stats as st
        f = self.fit
        return float(st.t.ppf(1 - alpha / 2, f.dfe)) if f.estdisp else float(st.norm.ppf(1 - alpha / 2))

    def m_predict(self, interp, args, nargout):
        pos, opts = get_opts(args, 0, ["Alpha", "BinomialSize", "Offset", "Simultaneous"], "predict")
        if len(pos) > 1:
            raise MatlabError("Too many input arguments for predict.")
        if "simultaneous" in opts and as_bool(opts["simultaneous"], "Simultaneous"):
            raise MatlabError("predict: simultaneous bounds are not supported for generalized linear models.")
        X = self._design_new(pos[0], "predict") if pos else self.Xd
        n = X.shape[0]
        off = np.zeros(n) if pos else self.offset[self.subset]
        if "offset" in opts:
            off = np.broadcast_to(vec(opts["offset"]), (n,)).astype(float)
        size = np.ones(n)
        if "binomialsize" in opts:
            if self.dist != "binomial":
                raise MatlabError("predict: BinomialSize applies to binomial models only.")
            size = np.broadcast_to(vec(opts["binomialsize"]), (n,)).astype(float)
        f = self.fit
        eta = X @ f.beta + off
        mu = self.link.inv(eta) * size
        if nargout < 2:
            return mu.reshape(-1, 1)
        alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
        half = self._crit(alpha) * np.sqrt(np.einsum("ij,jk,ik->i", X, f.cov, X))
        a, b = self.link.inv(eta - half) * size, self.link.inv(eta + half) * size
        return mu.reshape(-1, 1), np.column_stack([np.minimum(a, b), np.maximum(a, b)])

    def m_feval(self, interp, args, nargout):
        if not args:
            raise MatlabError("feval: the predictor values are missing.")
        if len(args) == 1:
            X = self._design_new(args[0], "feval")
            return self.link.inv(X @ self.fit.beta).reshape(-1, 1)
        if len(args) != len(self.md.preds):
            raise MatlabError(f"feval: expected {len(self.md.preds)} predictor arguments.")
        shape = np.asarray(fnum(args[0])).shape
        X = self._design_new(np.column_stack([vec(a) for a in args]), "feval")
        return self.link.inv(X @ self.fit.beta).reshape(shape, order="F")

    def m_random(self, interp, args, nargout):
        if len(args) > 1:
            raise MatlabError("Too many input arguments for random.")
        X = self._design_new(args[0], "random") if args else self.Xd
        mu = self.link.inv(X @ self.fit.beta)
        rng = rng_of(interp)
        s2 = self.fit.s ** 2
        if self.dist == "normal":
            r = mu + np.sqrt(s2) * rng.standard_normal(mu.size)
        elif self.dist == "binomial":
            r = rng.binomial(1, np.clip(mu, 0, 1)).astype(float)
        elif self.dist == "poisson":
            r = rng.poisson(mu).astype(float)
        elif self.dist == "gamma":
            r = rng.gamma(1 / s2, mu * s2)
        else:
            r = rng.wald(mu, 1 / s2)
        return r.reshape(-1, 1)

    def m_coefCI(self, interp, args, nargout):
        alpha = scalar(args[0], "alpha") if args else 0.05
        f = self.fit
        c = self._crit(alpha)
        return np.column_stack([f.beta - c * f.se, f.beta + c * f.se])

    def m_coefTest(self, interp, args, nargout):
        import scipy.stats as st
        f = self.fit
        p = len(self.cnames)
        H = np.atleast_2d(np.asarray(fnum(args[0]), dtype=float)) if args else \
            np.eye(p)[[i for i, o in enumerate(self.owner) if o >= 0]]
        if H.shape[1] != p:
            raise MatlabError(f"coefTest: H must have {p} columns.")
        c = vec(args[1]) if len(args) > 1 else np.zeros(H.shape[0])
        d = H @ f.beta - c
        V = H @ f.cov @ H.T
        r = int(np.linalg.matrix_rank(V))
        with np.errstate(all="ignore"):
            stat = float(d @ np.linalg.pinv(V) @ d)
        if f.estdisp:
            F = stat / r if r else np.nan
            return outs(nargout, float(st.f.sf(F, r, f.dfe)) if r else np.nan, F, float(r))
        return outs(nargout, float(st.chi2.sf(stat, r)) if r else np.nan, stat, float(r))

    def m_devianceTest(self, interp, args, nargout):
        import scipy.stats as st
        f, f0 = self.fit, self.null
        d = f0.dfe - f.dfe
        diff = f0.dev - f.dev
        with np.errstate(all="ignore"):
            if f.estdisp:
                stat = (diff / d) / f.s ** 2 if d else np.nan
                pv = float(st.f.sf(stat, d, f.dfe)) if d else np.nan
                sname = "FStat"
            else:
                stat = diff
                pv = float(st.chi2.sf(diff, d)) if d else np.nan
                sname = "chi2Stat"
        return make_table(["Deviance", "DFE", sname, "pValue"],
                          [np.array([f0.dev, f.dev]), np.array([float(f0.dfe), float(f.dfe)]),
                           np.array([np.nan, stat]), np.array([np.nan, pv])],
                          row_names=[formula_text(self.md, [], self.intercept, self.link.label()),
                                     formula_text(self.md, self.terms, self.intercept, self.link.label())])

    def m_plotResiduals(self, interp, args, nargout):
        kind = choice(args[0], ["histogram", "fitted", "caseorder", "probability"], "The plot type") if args \
            else "histogram"
        f = self.fit
        r = (f.y - f.mu) * f.N
        if kind == "histogram":
            h = builtin_call(interp, "histogram", [r.reshape(-1, 1)], 1)
        elif kind == "probability":
            h = builtin_call(interp, "normplot", [r.reshape(-1, 1)], 1)
        else:
            x = f.mu * f.N if kind == "fitted" else np.arange(1, r.size + 1.0)
            h = builtin_call(interp, "plot", [x.reshape(1, -1), r.reshape(1, -1), char("x")], 1)
        return h[0] if nargout else None

    def display_lines(self) -> list[str]:
        f = self.fit
        lines = ["", "Generalized linear regression model:",
                 "    " + formula_text(self.md, self.terms, self.intercept, self.link.label()),
                 f"    Distribution = {self.dist.title() if self.dist != 'inverse gaussian' else 'Inverse Gaussian'}",
                 "", "Estimated Coefficients:"]
        lines += _coef_lines(self.cnames, {"Estimate": f.beta, "SE": f.se, "tStat": f.t, "pValue": f.pvalues()})
        tbl = self.m_devianceTest(None, [], 1)
        stat = float(np.asarray(tbl.columns[2]).flat[1])
        pv = float(np.asarray(tbl.columns[3]).flat[1])
        lines += ["", "", f"{f.n} observations, {f.dfe} error degrees of freedom"]
        lines.append(f"Estimated Dispersion: {_fmt_g(f.s ** 2, 3)}" if f.estdisp else "Dispersion: 1")
        lines.append(f"{'F-statistic' if f.estdisp else 'Chi^2-statistic'} vs. constant model: {_fmt_g(stat, 3)}, "
                     f"p-value = {_fmt_g(pv, 3)}")
        return lines

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


_GLM_OPTS = ["BinomialSize", "CategoricalVars", "DispersionFlag", "Distribution", "Exclude", "Intercept", "Link",
             "Offset", "Options", "PredictorVars", "ResponseVar", "VarNames", "Weights", "B0"]


@builtin("fitglm")
def _fitglm(interp, args, nargout):
    pos, opts = get_opts(args, 1, _GLM_OPTS, "fitglm")
    md, terms, intercept, weights, exclude = _lm_common(pos, opts, "fitglm")
    dist = _dist_name(opts["distribution"]) if "distribution" in opts else "normal"
    link = Link(opts.get("link"), dist)
    offset = None
    if "offset" in opts:
        offset = np.broadcast_to(vec(opts["offset"]), (md.n,)).astype(float) if vec(opts["offset"]).size in (1, md.n) \
            else None
        if offset is None:
            raise MatlabError("fitglm: Offset must have one value per observation.")
    bsize = getattr(md, "trials", None)
    if "binomialsize" in opts:
        if dist != "binomial":
            raise MatlabError("fitglm: BinomialSize applies to the binomial distribution only.")
        bsize = np.broadcast_to(vec(opts["binomialsize"]), (md.n,)).astype(float)
    estdisp = as_bool(opts["dispersionflag"], "DispersionFlag") if "dispersionflag" in opts else None
    if estdisp is False and dist in ("normal", "gamma", "inverse gaussian"):
        estdisp = False
    return GeneralizedLinearModel(md, terms, intercept, dist, link, weights, exclude, offset, bsize, estdisp)


def _glm_xy(X_arg: Any, y_arg: Any, fname: str):
    X = np.asarray(fnum(X_arg), dtype=float)
    if X.ndim != 2:
        raise MatlabError(f"{fname}: X must be a matrix.")
    y = np.asarray(fnum(y_arg), dtype=float)
    N = None
    if y.ndim == 2 and y.shape[1] == 2 and y.shape[0] == X.shape[0] and X.shape[0] != 1:
        N = y[:, 1].copy()
        y = y[:, 0].copy()
    else:
        y = y.flatten(order="F")
    if y.size != X.shape[0]:
        raise MatlabError(f"{fname}: X and y must have the same number of rows.")
    return X, y, N


@builtin("glmfit")
def _glmfit(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("glmfit: expected glmfit(X, y, distr, 'Name', value, ...).")
    known = ["link", "constant", "offset", "weights", "estdisp", "options", "B0"]
    pos, opts = get_opts(args, 2, known, "glmfit")
    if len(pos) > 3:
        raise MatlabError("Too many input arguments for glmfit.")
    dist = _dist_name(pos[2]) if len(pos) > 2 and not is_empty_arg(pos[2]) else "normal"
    X, y, N = _glm_xy(pos[0], pos[1], "glmfit")
    const = True
    if "constant" in opts:
        const = as_bool(opts["constant"], "constant")
    link = Link(opts.get("link"), dist)
    n = X.shape[0]
    w = vec(opts["weights"]) if "weights" in opts and not is_empty_arg(opts["weights"]) else np.ones(n)
    off = np.broadcast_to(vec(opts["offset"]), (n,)).astype(float) if "offset" in opts and not is_empty_arg(
        opts["offset"]) else np.zeros(n)
    if w.size != n:
        raise MatlabError("glmfit: weights must have one value per observation.")
    estdisp = as_bool(opts["estdisp"], "estdisp") if "estdisp" in opts else None
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(w) | np.isnan(off))
    if N is not None:
        ok &= ~np.isnan(N)
    D = np.column_stack([np.ones(n), X]) if const else X
    if dist == "binomial":
        yy, NN = _binomial_response(y[ok], N[ok] if N is not None else None, "glmfit")
    else:
        yy, NN = y[ok], None
    f = GLMFit(D[ok], yy, dist, link, NN, w[ok], off[ok], estdisp)
    if nargout <= 1:
        return f.beta.reshape(-1, 1)

    def full(v):
        out = np.full(n, np.nan)
        out[ok] = v
        return out.reshape(-1, 1)

    with np.errstate(all="ignore"):
        sd = np.sqrt(np.diag(f.cov))
        corr = f.cov / np.outer(sd, sd)
        dres = np.sign(f.y - f.mu) * np.sqrt(np.clip(_deviance_terms(dist, f.y, f.mu, f.N), 0, None))
    stats = struct(beta=f.beta.reshape(-1, 1), dfe=float(f.dfe), sfit=f.sfit, s=float(f.s), estdisp=bool(f.estdisp),
                   covb=f.cov, se=f.se.reshape(-1, 1), coeffcorr=corr, t=f.t.reshape(-1, 1),
                   p=f.pvalues().reshape(-1, 1), resid=full((f.y - f.mu) * f.N), residp=full(f.pearson),
                   residd=full(dres), resida=full(_anscombe(dist, f.y, f.mu, f.N)), wts=full(f.W))
    return outs(nargout, f.beta.reshape(-1, 1), f.dev, stats)


def _anscombe(dist: str, y, mu, N):
    from scipy.special import betainc
    from scipy.special import beta as betafn
    with np.errstate(all="ignore"):
        if dist == "normal":
            return y - mu
        if dist == "poisson":
            return 1.5 * (y ** (2 / 3) - mu ** (2 / 3)) / mu ** (1 / 6)
        if dist == "gamma":
            return 3 * (y ** (1 / 3) - mu ** (1 / 3)) / mu ** (1 / 3)
        if dist == "inverse gaussian":
            return (np.log(y) - np.log(mu)) / np.sqrt(mu)
        t = 2.0 / 3.0
        b = betafn(t, t)
        return b * (betainc(t, t, y) - betainc(t, t, mu)) / ((mu * (1 - mu)) ** (1 / 6) / np.sqrt(N))


@builtin("glmval")
def _glmval(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 3:
        raise MatlabError("glmval: expected glmval(b, X, link, ...).")
    known = ["confidence", "size", "offset", "constant", "simultaneous"]
    pos, opts = get_opts(args, 3, known, "glmval")
    if len(pos) > 4:
        raise MatlabError("Too many input arguments for glmval.")
    b = vec(pos[0])
    X = np.asarray(fnum(pos[1]), dtype=float)
    link = Link(pos[2], "normal") if not is_empty_arg(pos[2]) else Link("identity", "normal")
    stats = pos[3] if len(pos) > 3 else None
    const = as_bool(opts["constant"], "constant") if "constant" in opts else True
    if X.ndim != 2:
        raise MatlabError("glmval: X must be a matrix.")
    if X.shape[1] != b.size - (1 if const else 0) and X.shape[0] == b.size - (1 if const else 0) and X.shape[1] == 1:
        X = X.T
    D = np.column_stack([np.ones(X.shape[0]), X]) if const else X
    if D.shape[1] != b.size:
        raise MatlabError("glmval: the number of coefficients does not match the number of predictors.")
    n = D.shape[0]
    off = np.broadcast_to(vec(opts["offset"]), (n,)).astype(float) if "offset" in opts else np.zeros(n)
    size = np.broadcast_to(vec(opts["size"]), (n,)).astype(float) if "size" in opts else np.ones(n)
    eta = D @ b + off
    yhat = link.inv(eta) * size
    if nargout < 2:
        return yhat.reshape(-1, 1)
    if not isinstance(stats, Struct):
        raise MatlabError("glmval: confidence bounds need the stats structure returned by glmfit.")
    if "simultaneous" in opts and as_bool(opts["simultaneous"], "simultaneous"):
        raise MatlabError("glmval: simultaneous bounds are not supported.")
    conf = scalar(opts["confidence"], "confidence") if "confidence" in opts else 0.95
    cov = np.asarray(fnum(stats.get_field(0, "covb")), dtype=float)
    dfe = scalar(stats.get_field(0, "dfe"))
    estdisp = bool(np.asarray(stats.get_field(0, "estdisp")).flat[0])
    crit = st.t.ppf(1 - (1 - conf) / 2, dfe) if estdisp else st.norm.ppf(1 - (1 - conf) / 2)
    half = crit * np.sqrt(np.einsum("ij,jk,ik->i", D, cov, D))
    a, c = link.inv(eta - half) * size, link.inv(eta + half) * size
    lo, hi = np.minimum(a, c), np.maximum(a, c)
    return yhat.reshape(-1, 1), (yhat - lo).reshape(-1, 1), (hi - yhat).reshape(-1, 1)
