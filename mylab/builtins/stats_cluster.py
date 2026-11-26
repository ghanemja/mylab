"""Distances, nearest neighbours and cluster analysis: pdist/pdist2, linkage trees, k-means,
k-medoids, DBSCAN, spectral clustering, Gaussian mixtures, silhouettes and cluster evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, Struct, char
from ._registry import builtin
from ._stats_util import (
    StatObject, as_bool, builtin_call, call, cellstr, choice, get_opts, given, group_codes, intval, is_empty_arg,
    outs, rng_of, scalar, struct, text_list, tied_rank, vec,
)
from ._util import fnum, is_text, text

_DIST_NAMES = ["euclidean", "squaredeuclidean", "seuclidean", "mahalanobis", "cityblock", "minkowski", "chebychev",
               "cosine", "correlation", "hamming", "jaccard", "spearman", "fasteuclidean", "fastsquaredeuclidean",
               "fastseuclidean"]
_DIST_ALIASES = {"sqeuclidean": "squaredeuclidean", "chebyshev": "chebychev", "fasteuclidean": "euclidean",
                 "fastsquaredeuclidean": "squaredeuclidean", "fastseuclidean": "seuclidean"}


def metric_name(v: Any, fname: str) -> Any:
    if isinstance(v, FunctionHandle):
        return v
    if not is_text(v):
        raise MatlabError(f"{fname}: the distance must be a name or a function handle.")
    s = text(v).lower()
    s = _DIST_ALIASES.get(s, s)
    if s not in _DIST_NAMES:
        hits = [d for d in _DIST_NAMES if d.startswith(s)]
        if len(hits) != 1:
            raise MatlabError(f"{fname}: unsupported distance '{text(v)}'.")
        s = _DIST_ALIASES.get(hits[0], hits[0])
    return s


def dist_matrix(interp, X: np.ndarray, Y: np.ndarray, metric: Any, extra: list, fname: str,
                ref: np.ndarray | None = None) -> np.ndarray:
    """Distances between the rows of X and the rows of Y (size nx-by-ny)."""
    from scipy.spatial.distance import cdist
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise MatlabError(f"{fname}: the two sets of points must have the same number of columns.")
    if isinstance(metric, FunctionHandle):
        D = np.empty((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            d = np.asarray(fnum(call(interp, metric, [X[i:i + 1], Y] + list(extra), 1)[0]), dtype=float).flatten()
            if d.size != Y.shape[0]:
                raise MatlabError(f"{fname}: the distance function must return one value per row of its second input.")
            D[i] = d
        return D
    metric = metric_name(metric, fname) if not isinstance(metric, str) else _DIST_ALIASES.get(metric, metric)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.zeros((X.shape[0], Y.shape[0]))
    ref = X if ref is None else ref
    param = extra[0] if extra and not is_empty_arg(extra[0]) else None
    with np.errstate(all="ignore"):
        if metric == "euclidean":
            return cdist(X, Y, "euclidean")
        if metric == "squaredeuclidean":
            return cdist(X, Y, "sqeuclidean")
        if metric == "seuclidean":
            s = vec(param) if param is not None else np.nanstd(ref, axis=0, ddof=1)
            if s.size != X.shape[1]:
                raise MatlabError(f"{fname}: the scale must have one value per column.")
            return cdist(X / s, Y / s, "euclidean")
        if metric == "mahalanobis":
            C = np.atleast_2d(np.asarray(fnum(param),
                    dtype=float)) if param is not None else np.atleast_2d(np.cov(ref, rowvar=False))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                raise MatlabError(f"{fname}: the covariance matrix must be symmetric and positive definite.") from None
            return cdist(np.linalg.solve(L, X.T).T, np.linalg.solve(L, Y.T).T, "euclidean")
        if metric == "cityblock":
            return cdist(X, Y, "cityblock")
        if metric == "chebychev":
            return cdist(X, Y, "chebyshev")
        if metric == "minkowski":
            p = scalar(param, "the Minkowski exponent") if param is not None else 2.0
            if p <= 0:
                raise MatlabError(f"{fname}: the Minkowski exponent must be positive.")
            return cdist(X, Y, "chebyshev") if np.isinf(p) else cdist(X, Y, "minkowski", p=p)
        if metric == "cosine":
            return cdist(X, Y, "cosine")
        if metric == "correlation":
            return cdist(X, Y, "correlation")
        if metric == "spearman":
            rx = np.vstack([tied_rank(r) for r in X])
            ry = np.vstack([tied_rank(r) for r in Y])
            return cdist(rx, ry, "correlation")
        if metric == "hamming":
            return (X[:, None, :] != Y[None, :, :]).mean(axis=2)
        if metric == "jaccard":
            ne = X[:, None, :] != Y[None, :, :]
            nz = (X[:, None, :] != 0) | (Y[None, :, :] != 0)
            return np.where(nz.sum(axis=2) > 0, (ne & nz).sum(axis=2) / np.maximum(nz.sum(axis=2), 1), 0.0)
    raise MatlabError(f"{fname}: unknown distance '{metric}'.")


def _condensed(D: np.ndarray) -> np.ndarray:
    return D[np.triu_indices(D.shape[0], 1)]


def _metric_and_extra(v: Any, fname: str) -> tuple[Any, list]:
    if isinstance(v, Cell):
        items = v.values_f()
        return metric_name(items[0], fname), items[1:]
    return metric_name(v, fname), []


@builtin("pdist")
def _pdist(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for pdist.")
    pos, opts = get_opts(args, 1, ["CacheSize"], "pdist")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError("pdist: X must be a matrix with one point per row.")
    metric = metric_name(pos[1], "pdist") if len(pos) > 1 else "euclidean"
    D = dist_matrix(interp, X, X, metric, list(pos[2:]), "pdist")
    if isinstance(metric, FunctionHandle):
        D = np.triu(D, 1) + np.triu(D, 1).T
    return _condensed(D).reshape(1, -1)


@builtin("pdist2")
def _pdist2(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("pdist2: expected pdist2(X, Y, distance, ...).")
    pos, opts = get_opts(args, 2, ["Smallest", "Largest", "SortIndices", "CacheSize", "Radius"], "pdist2")
    if "radius" in opts:
        raise MatlabError("pdist2: the option 'Radius' is not supported; use rangesearch.")
    X = np.asarray(fnum(pos[0]), dtype=float)
    Y = np.asarray(fnum(pos[1]), dtype=float)
    metric = metric_name(pos[2], "pdist2") if len(pos) > 2 else "euclidean"
    D = dist_matrix(interp, X, Y, metric, list(pos[3:]), "pdist2")
    if "smallest" not in opts and "largest" not in opts:
        if nargout > 1:
            raise MatlabError("pdist2 returns indices only with the 'Smallest' or 'Largest' option.")
        return D
    if "smallest" in opts and "largest" in opts:
        raise MatlabError("pdist2: give only one of 'Smallest' and 'Largest'.")
    key = "smallest" if "smallest" in opts else "largest"
    k = min(intval(opts[key], "number of distances"), D.shape[0])
    order = np.argsort(D if key == "smallest" else -D, axis=0, kind="stable")[:k, :]
    return np.take_along_axis(D, order, axis=0), order + 1.0


# -- nearest neighbours -----------------------------------------------------------------------------------

_NN_OPTS = ["K", "Distance", "P", "Cov", "Scale", "NSMethod", "IncludeTies", "BucketSize", "SortIndices"]


def _nn_extra(opts: dict, metric: Any) -> list:
    if metric == "minkowski" and "p" in opts:
        return [opts["p"]]
    if metric == "mahalanobis" and "cov" in opts:
        return [opts["cov"]]
    if metric == "seuclidean" and "scale" in opts:
        return [opts["scale"]]
    return []


def knn_core(interp, X, Y, opts: dict, fname: str, default_metric: Any = "euclidean"):
    k = intval(opts["k"], "K") if "k" in opts else 1
    if k < 1:
        raise MatlabError(f"{fname}: K must be a positive integer.")
    metric = metric_name(opts["distance"], fname) if "distance" in opts else default_metric
    if "nsmethod" in opts:
        choice(opts["nsmethod"], ["kdtree", "exhaustive"], "NSMethod")   # both give the exact neighbours
    ties = as_bool(opts["includeties"], "IncludeTies") if "includeties" in opts else False
    D = dist_matrix(interp, Y, X, metric, _nn_extra(opts, metric), fname, ref=X)
    kk = min(k, X.shape[0])
    order = np.argsort(D, axis=1, kind="stable")
    if not ties:
        idx = order[:, :kk]
        return idx + 1.0, np.take_along_axis(D, idx, axis=1)
    ci, cd = [], []
    for i in range(D.shape[0]):
        d = D[i, order[i]]
        cut = d[kk - 1] if kk else -np.inf
        m = int(np.sum(d <= cut))
        ci.append((order[i, :m] + 1.0).reshape(1, -1))
        cd.append(d[:m].reshape(1, -1))
    return Cell.from_list(ci, column=True), Cell.from_list(cd, column=True)


@builtin("knnsearch")
def _knnsearch(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("knnsearch: expected knnsearch(X, Y, 'Name', value, ...).")
    if isinstance(args[0], Searcher):
        return args[0].m_knnsearch(interp, list(args[1:]), nargout)
    pos, opts = get_opts(args, 2, _NN_OPTS, "knnsearch")
    if len(pos) != 2:
        raise MatlabError("knnsearch: expected knnsearch(X, Y, 'Name', value, ...).")
    X, Y = np.asarray(fnum(pos[0]), dtype=float), np.asarray(fnum(pos[1]), dtype=float)
    return outs(nargout, *knn_core(interp, X, Y, opts, "knnsearch"))


def range_core(interp, X, Y, r: float, opts: dict, fname: str, default_metric: Any = "euclidean"):
    metric = metric_name(opts["distance"], fname) if "distance" in opts else default_metric
    D = dist_matrix(interp, Y, X, metric, _nn_extra(opts, metric), fname, ref=X)
    sort = as_bool(opts["sortindices"], "SortIndices") if "sortindices" in opts else True
    ci, cd = [], []
    for i in range(D.shape[0]):
        hit = np.flatnonzero(D[i] <= r)
        if sort:
            hit = hit[np.argsort(D[i, hit], kind="stable")]
        ci.append((hit + 1.0).reshape(1, -1))
        cd.append(D[i, hit].reshape(1, -1))
    return Cell.from_list(ci, column=True), Cell.from_list(cd, column=True)


@builtin("rangesearch")
def _rangesearch(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("rangesearch: expected rangesearch(X, Y, r, 'Name', value, ...).")
    if isinstance(args[0], Searcher):
        return args[0].m_rangesearch(interp, list(args[1:]), nargout)
    pos, opts = get_opts(args, 3, _NN_OPTS[1:], "rangesearch")
    if len(pos) != 3:
        raise MatlabError("rangesearch: expected rangesearch(X, Y, r, 'Name', value, ...).")
    X, Y = np.asarray(fnum(pos[0]), dtype=float), np.asarray(fnum(pos[1]), dtype=float)
    return outs(nargout, *range_core(interp, X, Y, scalar(pos[2], "r"), opts, "rangesearch"))


class Searcher(StatObject):
    """KDTreeSearcher / ExhaustiveSearcher: training points with a default distance."""

    def __init__(self, kind: str, X: np.ndarray, metric: Any, params: dict, bucket: float):
        super().__init__()
        self.matlab_class = kind
        self.X, self.metric, self.params, self.bucket = X, metric, params, bucket

    def property_names(self) -> list[str]:
        return ["X", "Distance", "DistParameter"] + (["BucketSize"] if self.matlab_class == "KDTreeSearcher" else [])

    def get_field(self, interp, name: str) -> Any:
        if name == "X":
            return self.X
        if name == "Distance":
            return self.metric if isinstance(self.metric, FunctionHandle) else char(self.metric)
        if name == "DistParameter":
            return next(iter(self.params.values()), np.zeros((0, 0)))
        if name == "BucketSize" and self.matlab_class == "KDTreeSearcher":
            return float(self.bucket)
        return super().get_field(interp, name)

    def _opts(self, args: list, n: int, known: list, fname: str):
        pos, opts = get_opts(args, n, known, fname)
        if len(pos) != n:
            raise MatlabError(f"{fname}: wrong number of input arguments.")
        for k, v in self.params.items():
            opts.setdefault(k, v)
        return pos, opts

    def m_knnsearch(self, interp, args, nargout):
        pos, opts = self._opts(args, 1, ["K", "Distance", "P", "Cov", "Scale", "IncludeTies", "SortIndices"],
                               "knnsearch")
        Y = np.asarray(fnum(pos[0]), dtype=float)
        return outs(nargout, *knn_core(interp, self.X, Y, opts, "knnsearch", self.metric))

    def m_rangesearch(self, interp, args, nargout):
        pos, opts = self._opts(args, 2, ["Distance", "P", "Cov", "Scale", "SortIndices"], "rangesearch")
        Y = np.asarray(fnum(pos[0]), dtype=float)
        return outs(nargout, *range_core(interp, self.X, Y, scalar(pos[1], "r"), opts, "rangesearch", self.metric))

    def display_lines(self) -> list[str]:
        d = repr(self.metric) if isinstance(self.metric, FunctionHandle) else f"'{self.metric}'"
        return [f"  {self.matlab_class} with properties:", "", f"       Distance: {d}",
                f"              X: [{self.X.shape[0]}x{self.X.shape[1]} double]"]


def _make_searcher(interp, kind: str, args: list) -> Searcher:
    pos, opts = get_opts(args, 1, ["Distance", "P", "Cov", "Scale", "BucketSize", "NSMethod"], kind)
    if len(pos) != 1:
        raise MatlabError(f"{kind}: expected {kind}(X, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError(f"{kind}: X must be a matrix with one point per row.")
    metric = metric_name(opts["distance"], kind) if "distance" in opts else "euclidean"
    if kind == "KDTreeSearcher" and metric not in ("euclidean", "cityblock", "chebychev", "minkowski"):
        raise MatlabError("KDTreeSearcher supports the euclidean, cityblock, chebychev and minkowski distances.")
    params = {k: opts[k] for k in ("p", "cov", "scale") if k in opts}
    bucket = scalar(opts["bucketsize"], "BucketSize") if "bucketsize" in opts else 50.0
    return Searcher(kind, X, metric, params, bucket)


@builtin("KDTreeSearcher")
def _kdtreesearcher(interp, args, nargout):
    return _make_searcher(interp, "KDTreeSearcher", list(args))


@builtin("ExhaustiveSearcher")
def _exhaustivesearcher(interp, args, nargout):
    return _make_searcher(interp, "ExhaustiveSearcher", list(args))


@builtin("createns")
def _createns(interp, args, nargout):
    pos, opts = get_opts(args, 1, ["Distance", "P", "Cov", "Scale", "BucketSize", "NSMethod"], "createns")
    if len(pos) != 1:
        raise MatlabError("createns: expected createns(X, 'Name', value, ...).")
    if "nsmethod" in opts:
        kind = choice(opts["nsmethod"], ["kdtree", "exhaustive"], "NSMethod")
    else:
        m = metric_name(opts["distance"], "createns") if "distance" in opts else "euclidean"
        ncol = np.asarray(fnum(pos[0])).shape[1]
        kind = "kdtree" if m in ("euclidean", "cityblock", "chebychev", "minkowski") and ncol <= 10 else "exhaustive"
    rest = [pos[0]]
    i = 1
    while i < len(args):                      # pass the options on, without NSMethod
        if not (is_text(args[i]) and text(args[i]).lower() == "nsmethod"):
            rest += [args[i], args[i + 1]]
        i += 2
    return _make_searcher(interp, "KDTreeSearcher" if kind == "kdtree" else "ExhaustiveSearcher", rest)


# -- hierarchical clustering ----------------------------------------------------------------------------------

_LINK_METHODS = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]


def _is_dist_vector(a: np.ndarray) -> bool:
    if a.ndim != 2 or a.shape[0] != 1 or a.shape[1] < 1:
        return False
    n = (1 + np.sqrt(1 + 8 * a.shape[1])) / 2
    return n == int(n)


def linkage_core(interp, data: Any, method: str, metric_arg: Any, fname: str = "linkage") -> np.ndarray:
    from scipy.cluster.hierarchy import linkage
    A = np.asarray(fnum(data), dtype=float)
    if _is_dist_vector(A):
        y = A.flatten()
        if metric_arg is not None:
            raise MatlabError(f"{fname}: a distance name cannot be given together with a distance vector.")
    else:
        if A.ndim != 2 or A.shape[0] < 2:
            raise MatlabError(f"{fname}: the data must have at least two rows.")
        metric, extra = _metric_and_extra(metric_arg, fname) if metric_arg is not None else ("euclidean", [])
        y = _condensed(dist_matrix(interp, A, A, metric, extra, fname))
    if np.isnan(y).any() or np.any(y < 0):
        raise MatlabError(f"{fname}: distances must be non-negative and must not be NaN.")
    Z = linkage(y, method=method)
    out = np.empty((Z.shape[0], 3))
    out[:, 0] = np.minimum(Z[:, 0], Z[:, 1]) + 1
    out[:, 1] = np.maximum(Z[:, 0], Z[:, 1]) + 1
    out[:, 2] = Z[:, 2]
    return out


@builtin("linkage")
def _linkage(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for linkage.")
    pos, opts = get_opts(args, 1, ["savememory"], "linkage")
    if "savememory" in opts:
        as_bool(opts["savememory"], "savememory")   # a memory/speed switch: the result is the same either way
    if len(pos) > 3:
        raise MatlabError("Too many input arguments for linkage.")
    method = choice(pos[1], _LINK_METHODS, "The linkage method") if len(pos) > 1 and not is_empty_arg(pos[1]) \
        else "single"
    return linkage_core(interp, pos[0], method, pos[2] if len(pos) > 2 else None)


def _check_tree(Z_arg: Any, fname: str) -> np.ndarray:
    Z = np.asarray(fnum(Z_arg), dtype=float)
    if Z.ndim != 2 or Z.shape[1] != 3 or Z.shape[0] < 1:
        raise MatlabError(f"{fname}: Z must be a tree from linkage (three columns).")
    return Z


def _scipy_tree(Z: np.ndarray) -> np.ndarray:
    """SciPy's four-column form (0-based children, heights, leaf counts)."""
    m = Z.shape[0] + 1
    counts = np.ones(2 * m - 1)
    out = np.empty((m - 1, 4))
    for i in range(m - 1):
        a, b = int(Z[i, 0]) - 1, int(Z[i, 1]) - 1
        counts[m + i] = counts[a] + counts[b]
        out[i] = (a, b, Z[i, 2], counts[m + i])
    return out


def _labels_first_seen(roots: np.ndarray) -> np.ndarray:
    seen: dict = {}
    out = np.empty(roots.size)
    for i, r in enumerate(roots):
        if r not in seen:
            seen[r] = len(seen) + 1.0
        out[i] = seen[r]
    return out


def _cut(Z: np.ndarray, join: np.ndarray) -> np.ndarray:
    """Cluster numbers (in order of first appearance) after merging the rows flagged in ``join``."""
    m = Z.shape[0] + 1
    parent = np.arange(2 * m - 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m - 1):
        if join[i]:
            node = m + i
            parent[find(int(Z[i, 0]) - 1)] = node
            parent[find(int(Z[i, 1]) - 1)] = node
    return _labels_first_seen(np.array([find(i) for i in range(m)]))


def _subtree_max(Z: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Largest value of ``vals`` over each node and the nodes below it."""
    m = Z.shape[0] + 1
    out = vals.astype(float).copy()
    for i in range(m - 1):
        for c in (int(Z[i, 0]) - 1, int(Z[i, 1]) - 1):
            if c >= m:
                out[i] = max(out[i], out[c - m])
    return out


def cluster_tree(Z: np.ndarray, opts: dict, fname: str) -> np.ndarray:
    from scipy.cluster.hierarchy import inconsistent
    m = Z.shape[0] + 1
    cols = []
    if "maxclust" in opts:
        if "cutoff" in opts:
            raise MatlabError(f"{fname}: give either 'Cutoff' or 'MaxClust', not both.")
        top = _subtree_max(Z, Z[:, 2])
        for k in vec(opts["maxclust"]):
            if k < 1 or k != int(k):
                raise MatlabError(f"{fname}: MaxClust must be a positive integer.")
            thresholds = np.unique(top)
            join = np.zeros(m - 1, dtype=bool)
            for t in thresholds:              # the lowest cut that leaves at most k clusters
                join = top <= t
                if m - int(join.sum()) <= k:
                    break
            if m <= k:
                join = np.zeros(m - 1, dtype=bool)
            cols.append(_cut(Z, join))
    elif "cutoff" in opts:
        crit = choice(opts["criterion"], ["inconsistent", "distance"], "Criterion") if "criterion" in opts \
            else "inconsistent"
        depth = intval(opts["depth"], "Depth") if "depth" in opts else 2
        if crit == "inconsistent":
            vals = _subtree_max(Z, inconsistent(_scipy_tree(Z), depth)[:, 3])
        else:
            if "depth" in opts:
                raise MatlabError(f"{fname}: 'Depth' applies to the 'inconsistent' criterion only.")
            vals = _subtree_max(Z, Z[:, 2])
        for c in vec(opts["cutoff"]):
            cols.append(_cut(Z, vals < c))
    else:
        raise MatlabError(f"{fname}: give 'Cutoff' or 'MaxClust'.")
    return np.column_stack(cols)


@builtin("cluster")
def _cluster(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("cluster: expected cluster(Z, 'Cutoff', c) or cluster(Z, 'MaxClust', n).")
    Z = _check_tree(args[0], "cluster")
    if not is_text(args[1]):            # older form: cluster(Z, cutoff) / cluster(Z, cutoff, depth)
        c = vec(args[1])
        opts: dict = {"maxclust": args[1]} if np.all((c >= 2) & (c == np.round(c))) else {"cutoff": args[1]}
        if len(args) == 3 and "cutoff" in opts:
            opts["depth"] = args[2]
        elif len(args) > 2:
            raise MatlabError("cluster: expected cluster(Z, 'Cutoff', c) or cluster(Z, 'MaxClust', n).")
    else:
        _, opts = get_opts(args, 1, ["Cutoff", "MaxClust", "Criterion", "Depth"], "cluster")
    return cluster_tree(Z, opts, "cluster")


@builtin("clusterdata")
def _clusterdata(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("clusterdata: expected clusterdata(X, cutoff) or clusterdata(X, 'Name', value, ...).")
    known = ["Cutoff", "MaxClust", "Criterion", "Depth", "Distance", "Linkage", "SaveMemory"]
    if not is_text(args[1]):
        c = scalar(args[1], "cutoff")
        pos, opts = get_opts(args, 2, known, "clusterdata")
        opts["maxclust" if (c >= 2 and c == int(c)) else "cutoff"] = args[1]
    else:
        pos, opts = get_opts(args, 1, known, "clusterdata")
    method = choice(opts["linkage"], _LINK_METHODS, "Linkage") if "linkage" in opts else "single"
    Z = linkage_core(interp, pos[0], method, opts.get("distance", char("euclidean")), "clusterdata")
    return cluster_tree(Z, {k: v for k, v in opts.items() if k in ("cutoff", "maxclust", "criterion", "depth")},
                        "clusterdata")


@builtin("inconsistent")
def _inconsistent(interp, args, nargout):
    from scipy.cluster.hierarchy import inconsistent
    if not 1 <= len(args) <= 2:
        raise MatlabError("inconsistent: expected inconsistent(Z) or inconsistent(Z, depth).")
    Z = _check_tree(args[0], "inconsistent")
    d = intval(args[1], "depth") if given(args, 1) else 2
    return inconsistent(_scipy_tree(Z), d)


@builtin("cophenet")
def _cophenet(interp, args, nargout):
    from scipy.cluster.hierarchy import cophenet
    if not 1 <= len(args) <= 2:
        raise MatlabError("cophenet: expected cophenet(Z, Y).")
    Z = _check_tree(args[0], "cophenet")
    d = cophenet(_scipy_tree(Z))
    if len(args) < 2:
        return outs(nargout, np.nan, d.reshape(1, -1)) if nargout > 1 else d.reshape(1, -1)
    y = vec(args[1])
    if y.size != d.size:
        raise MatlabError("cophenet: Y must be the distance vector the tree was built from.")
    with np.errstate(all="ignore"):
        c = float(np.corrcoef(y, d)[0, 1])
    return outs(nargout, c, d.reshape(1, -1))


@builtin("optimalleaforder")
def _optimalleaforder(interp, args, nargout):
    from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
    if len(args) < 2:
        raise MatlabError("optimalleaforder: expected optimalleaforder(tree, D).")
    pos, opts = get_opts(args, 2, ["Criteria", "Transformation"], "optimalleaforder")
    if opts:
        raise MatlabError("optimalleaforder: the options 'Criteria' and 'Transformation' are not supported.")
    Z = _check_tree(pos[0], "optimalleaforder")
    D = np.asarray(fnum(pos[1]), dtype=float)
    y = _condensed(D) if D.ndim == 2 and D.shape[0] == D.shape[1] and D.shape[0] > 1 else D.flatten()
    return (leaves_list(optimal_leaf_ordering(_scipy_tree(Z), y)) + 1.0).reshape(1, -1)


@builtin("dendrogram")
def _dendrogram(interp, args, nargout):
    from scipy.cluster.hierarchy import dendrogram
    if not args:
        raise MatlabError("Not enough input arguments for dendrogram.")
    pos, opts = get_opts(args, 1, ["Orientation", "Labels", "ColorThreshold", "Reorder", "CheckCrossing"],
                         "dendrogram")
    Z = _check_tree(pos[0], "dendrogram")
    m = Z.shape[0] + 1
    P = intval(pos[1], "P") if len(pos) > 1 and not is_empty_arg(pos[1]) else 30
    if len(pos) > 2:
        raise MatlabError("Too many input arguments for dendrogram.")
    if P == 0 or P >= m:
        P = m
    orient = choice(opts["orientation"], ["top", "bottom", "left", "right"], "Orientation") \
        if "orientation" in opts else "top"
    if "reorder" in opts:
        raise MatlabError("dendrogram: the option 'Reorder' is not supported.")
    Zs = _scipy_tree(Z)
    kw: dict = {"no_plot": True}
    if P < m:
        kw.update(truncate_mode="lastp", p=P)
    thr = None
    if "colorthreshold" in opts:
        v = opts["colorthreshold"]
        thr = 0.7 * Z[:, 2].max() if is_text(v) else scalar(v, "ColorThreshold")
    kw["color_threshold"] = thr if thr is not None else 0
    info = dendrogram(Zs, **kw)
    leaves = info["leaves"]
    if P < m:
        T = cluster_tree(Z, {"maxclust": np.array([[float(P)]])}, "dendrogram")[:, 0]
        node_of_leaf = {}
        members: dict = {i: [i] for i in range(m)}
        for i in range(m - 1):
            members[m + i] = members[int(Zs[i, 0])] + members[int(Zs[i, 1])]
        for lf in leaves:
            node_of_leaf[lf] = T[members[lf][0]]
        outperm = np.array([node_of_leaf[lf] for lf in leaves])
        names = [str(int(v)) for v in outperm]
    else:
        T = np.arange(1.0, m + 1)
        outperm = np.array(leaves, dtype=float) + 1
        names = [str(int(v)) for v in outperm]
    if "labels" in opts:
        lab = text_list(opts["labels"])
        if lab is None or len(lab) != m:
            raise MatlabError("dendrogram: Labels must give one label per observation.")
        if P >= m:
            names = [lab[int(v) - 1] for v in outperm]
    hs = []
    colors = info.get("color_list", [])
    for k, (ic, dc) in enumerate(zip(info["icoord"], info["dcoord"])):
        xs = (np.array(ic) + 5.0) / 10.0
        ys = np.array(dc)
        if orient in ("left", "right"):
            xs, ys = ys, xs
        col = colors[k] if thr is not None and k < len(colors) else "C0"
        rgb = _mpl_rgb(col) if thr is not None else np.array([[0.0, 0.447, 0.741]])
        h = builtin_call(interp, "plot", [xs.reshape(1, -1), ys.reshape(1, -1), char("Color"), rgb], 1)
        hs.append(h[0])
        builtin_call(interp, "hold", [char("on")], 0)
    builtin_call(interp, "hold", [char("off")], 0)
    gca = builtin_call(interp, "gca", [], 1)[0]
    ticks = np.arange(1.0, len(leaves) + 1).reshape(1, -1)
    axis_name = "XTick" if orient in ("top", "bottom") else "YTick"
    builtin_call(interp, "set", [gca, char(axis_name), ticks, char(axis_name + "Label"), cellstr(names,
            column=False)], 0)
    lim = char("XLim" if orient in ("top", "bottom") else "YLim")
    builtin_call(interp, "set", [gca, lim, np.array([[0.0, len(leaves) + 1.0]])], 0)
    if orient == "bottom":
        builtin_call(interp, "set", [gca, char("YDir"), char("reverse")], 0)
    if orient == "right":
        builtin_call(interp, "set", [gca, char("XDir"), char("reverse")], 0)
    if not nargout:
        return None
    from .plotting import GraphicsArray
    return outs(nargout, hs[0] if len(hs) == 1 else GraphicsArray(hs), T.reshape(-1, 1), outperm.reshape(1, -1))


def _mpl_rgb(c: str) -> np.ndarray:
    from matplotlib.colors import to_rgb
    return np.array([list(to_rgb(c))])


# -- k-means --------------------------------------------------------------------------------------------------

_KM_DIST = ["sqeuclidean", "cityblock", "cosine", "correlation", "hamming"]


def _km_prepare(X: np.ndarray, dist: str, fname: str) -> np.ndarray:
    if dist == "cosine":
        nrm = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        if np.any(nrm == 0):
            raise MatlabError(f"{fname}: the cosine distance is undefined for points of zero length.")
        return X / nrm
    if dist == "correlation":
        Xc = X - X.mean(axis=1, keepdims=True)
        nrm = np.sqrt(np.sum(Xc ** 2, axis=1, keepdims=True))
        if np.any(nrm == 0):
            raise MatlabError(f"{fname}: the correlation distance is undefined for constant points.")
        return Xc / nrm
    if dist == "hamming" and not np.all((X == 0) | (X == 1)):
        raise MatlabError(f"{fname}: the hamming distance needs binary (0/1) data.")
    return X


def _km_dist(X: np.ndarray, C: np.ndarray, dist: str) -> np.ndarray:
    if dist == "sqeuclidean":
        d = np.sum(X ** 2, axis=1)[:, None] + np.sum(C ** 2, axis=1)[None, :] - 2 * X @ C.T
        return np.clip(d, 0, None)
    if dist == "cityblock":
        return np.abs(X[:, None, :] - C[None]).sum(axis=2)
    if dist in ("cosine", "correlation"):
        return np.clip(1 - X @ C.T, 0, None)
    return np.abs(X[:, None, :] - C[None]).mean(axis=2)


def _km_centroid(M: np.ndarray, dist: str) -> np.ndarray:
    if dist == "cityblock":
        return np.median(M, axis=0)
    if dist == "hamming":
        return (M.mean(axis=0) > 0.5).astype(float) if M.shape[0] % 2 else (np.median(M, axis=0) > 0.5).astype(float)
    c = M.mean(axis=0)
    if dist in ("cosine", "correlation"):
        if dist == "correlation":
            c = c - c.mean()
        nrm = np.sqrt(np.sum(c ** 2))
        return c / nrm if nrm > 0 else c
    return c


def _km_seed(rng, X: np.ndarray, k: int, dist: str, mode: str) -> np.ndarray:
    n, p = X.shape
    if mode == "sample":
        return X[rng.choice(n, k, replace=False)].copy()
    if mode == "uniform":
        lo, hi = X.min(axis=0), X.max(axis=0)
        return lo + (hi - lo) * rng.random((k, p))
    if mode == "cluster":
        sub = X[rng.choice(n, max(k, int(np.ceil(0.1 * n))), replace=False)]
        return kmeans_core(rng, sub, k, dist, [_km_seed(rng, sub, k, dist, "sample")], 100, "singleton")[1]
    C = [X[rng.integers(n)]]
    d2 = _km_dist(X, np.array(C), dist)[:, 0]
    for _ in range(1, k):
        tot = d2.sum()
        C.append(X[rng.choice(n, p=d2 / tot)] if tot > 0 else X[rng.integers(n)])
        d2 = np.minimum(d2, _km_dist(X, np.array(C[-1:]), dist)[:, 0])
    return np.array(C)


def kmeans_core(rng, X: np.ndarray, k: int, dist: str, starts: list, max_iter: int, empty: str):
    """Lloyd iterations from each start; returns (idx, C, sumd, D, iterations) of the best run."""
    n = X.shape[0]
    best = None
    for C0 in starts:
        C = C0.astype(float).copy()
        idx = np.full(n, -1)
        it = 0
        for it in range(1, max_iter + 1):
            D = _km_dist(X, C, dist)
            new = np.argmin(D, axis=1)
            for j in range(k):
                if not np.any(new == j):
                    if empty == "error":
                        raise MatlabError(f"kmeans: an empty cluster was created at iteration {it}.")
                    if empty == "singleton":
                        far = int(np.argmax(D[np.arange(n), new]))
                        new[far] = j
                        D[far, j] = 0.0
            if np.array_equal(new, idx):
                break
            idx = new
            for j in range(k):
                members = X[idx == j]
                C[j] = _km_centroid(members, dist) if members.size else np.nan
        D = _km_dist(X, C, dist)
        if empty == "drop":
            D = np.where(np.isnan(D), np.inf, D)
        idx = np.argmin(D, axis=1)
        sumd = np.array([D[idx == j, j].sum() for j in range(k)])
        if best is None or sumd.sum() < best[2].sum():
            best = (idx, C, sumd, D, it)
    return best


@builtin("kmeans")
def _kmeans(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("kmeans: expected kmeans(X, k, 'Name', value, ...).")
    known = ["Distance", "Start", "Replicates", "MaxIter", "EmptyAction", "Display", "OnlinePhase", "Options"]
    pos, opts = get_opts(args, 2, known, "kmeans")
    if len(pos) != 2:
        raise MatlabError("kmeans: expected kmeans(X, k, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError("kmeans: X must be a matrix.")
    k = intval(pos[1], "number of clusters")
    dist = choice(opts["distance"], _KM_DIST, "Distance") if "distance" in opts else "sqeuclidean"
    max_iter = intval(opts["maxiter"], "MaxIter") if "maxiter" in opts else 100
    o = opts.get("options")
    if isinstance(o, Struct) and "MaxIter" in o.fields and not is_empty_arg(o.get_field(0, "MaxIter")):
        max_iter = intval(o.get_field(0, "MaxIter"), "MaxIter")
    elif o is not None and not is_empty_arg(o) and not isinstance(o, Struct):
        raise MatlabError("kmeans: Options must be a structure created by statset.")
    empty = choice(opts["emptyaction"], ["singleton", "error", "drop"], "EmptyAction") if "emptyaction" in opts \
        else "singleton"
    display = choice(opts["display"], ["off", "final", "iter"], "Display") if "display" in opts else "off"
    if "onlinephase" in opts and as_bool(opts["onlinephase"], "OnlinePhase"):
        raise MatlabError("kmeans: the online update phase ('OnlinePhase', 'on') is not supported.")
    keep = ~np.isnan(X).any(axis=1)
    Xk = _km_prepare(X[keep], dist, "kmeans")
    n, p = Xk.shape
    if k < 1 or k > n:
        raise MatlabError("kmeans: number of clusters must be between 1 and the number of points.")
    rng = rng_of(interp)
    start = opts.get("start")
    if start is not None and not is_text(start):
        S = np.asarray(fnum(start), dtype=float)
        if S.shape[:2] != (k, X.shape[1]):
            raise MatlabError("kmeans: Start matrix must be k-by-p.")
        starts = [S[:, :, r] for r in range(S.shape[2])] if S.ndim == 3 else [S]
        if "replicates" in opts and intval(opts["replicates"], "Replicates") != len(starts):
            raise MatlabError("kmeans: Replicates must match the number of pages of the Start array.")
        starts = [_km_prepare(s, dist, "kmeans") if dist in ("cosine", "correlation") else s for s in starts]
    else:
        mode = choice(start, ["plus", "sample", "uniform", "cluster"], "Start") if start is not None else "plus"
        reps = intval(opts["replicates"], "Replicates") if "replicates" in opts else 1
        if reps < 1:
            raise MatlabError("kmeans: Replicates must be a positive integer.")
        starts = [_km_seed(rng, Xk, k, dist, mode) for _ in range(reps)]
    idx, C, sumd, D, iters = kmeans_core(rng, Xk, k, dist, starts, max_iter, empty)
    if display != "off":
        interp.out(f"Best total sum of distances = {sumd.sum():g}\n")
    idx_full = np.full((X.shape[0], 1), np.nan)
    idx_full[keep, 0] = idx + 1.0
    D_full = np.full((X.shape[0], k), np.nan)
    D_full[keep] = D
    return outs(nargout, idx_full, C, sumd.reshape(-1, 1), D_full)


# -- k-medoids ---------------------------------------------------------------------------------------------------

def _pam(rng, D: np.ndarray, k: int, med0: np.ndarray, max_iter: int = 200) -> tuple[np.ndarray, int]:
    med = med0.copy()
    it = 0
    while it < max_iter:
        it += 1
        idx = np.argmin(D[:, med], axis=1)
        new = med.copy()
        for j in range(k):
            members = np.flatnonzero(idx == j)
            if members.size:
                new[j] = members[np.argmin(D[np.ix_(members, members)].sum(axis=0))]
        if np.array_equal(new, med):
            break
        med = new
    # swap phase: exchange a medoid with a non-medoid while that lowers the total distance
    n = D.shape[0]
    if n <= 2000:
        improved = True
        while improved:
            improved = False
            cur = np.min(D[:, med], axis=1).sum()
            for j in range(k):
                others = np.delete(med, j)
                base = np.min(D[:, others], axis=1) if others.size else np.full(n, np.inf)
                tot = np.minimum(base[:, None], D).sum(axis=0)
                tot[med] = np.inf
                c = int(np.argmin(tot))
                if tot[c] < cur - 1e-12 * max(cur, 1.0):
                    med[j] = c
                    cur = tot[c]
                    improved = True
    return med, it


@builtin("kmedoids")
def _kmedoids(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("kmedoids: expected kmedoids(X, k, 'Name', value, ...).")
    known = ["Algorithm", "Distance", "Start", "Replicates", "Options", "OnlinePhase", "NumSamples",
             "PercentNeighbors"]
    pos, opts = get_opts(args, 2, known, "kmedoids")
    if len(pos) != 2:
        raise MatlabError("kmedoids: expected kmedoids(X, k, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    k = intval(pos[1], "k")
    algo = choice(opts["algorithm"], ["pam", "small", "clara", "large"], "Algorithm") if "algorithm" in opts else "pam"
    metric = metric_name(opts["distance"], "kmedoids") if "distance" in opts else "squaredeuclidean"
    keep = ~np.isnan(X).any(axis=1)
    Xk = X[keep]
    n = Xk.shape[0]
    if k < 1 or k > n:
        raise MatlabError("kmedoids: k must be between 1 and the number of points.")
    D = dist_matrix(interp, Xk, Xk, metric, [], "kmedoids")
    rng = rng_of(interp)
    reps = intval(opts["replicates"], "Replicates") if "replicates" in opts else 1
    start = opts.get("start")
    best = None
    for r in range(reps):
        if start is not None and not is_text(start):
            S = np.asarray(fnum(start), dtype=float)
            if S.shape != (k, X.shape[1]):
                raise MatlabError("kmedoids: Start must be a k-by-p matrix of data points.")
            med0 = np.argmin(dist_matrix(interp, S, Xk, "squaredeuclidean", [], "kmedoids"), axis=1)
        else:
            mode = choice(start, ["plus", "sample", "cluster"], "Start") if start is not None else "plus"
            if mode == "sample":
                med0 = rng.choice(n, k, replace=False)
            else:
                med0 = [int(rng.integers(n))]
                d = D[:, med0[0]].copy()
                for _ in range(1, k):
                    tot = d.sum()
                    nxt = int(rng.choice(n, p=d / tot)) if tot > 0 else int(rng.integers(n))
                    med0.append(nxt)
                    d = np.minimum(d, D[:, nxt])
                med0 = np.array(med0)
        med, it = _pam(rng, D, k, np.asarray(med0))
        tot = np.min(D[:, med], axis=1).sum()
        if best is None or tot < best[0]:
            best = (tot, med, it, r + 1)
    _, med, it, rep = best
    Dk = D[:, med]
    idx = np.argmin(Dk, axis=1)
    sumd = np.array([Dk[idx == j, j].sum() for j in range(k)])
    idx_full = np.full((X.shape[0], 1), np.nan)
    idx_full[keep, 0] = idx + 1.0
    D_full = np.full((X.shape[0], k), np.nan)
    D_full[keep] = Dk
    midx = np.flatnonzero(keep)[med] + 1.0
    info = struct(algorithm=algo, start="plus" if start is None or not is_text(start) else text(start),
                  distance=metric if isinstance(metric,
                          str) else "custom", iterations=float(it), bestReplicate=float(rep))
    return outs(nargout, idx_full, Xk[med], sumd.reshape(-1, 1), D_full, midx.reshape(-1, 1), info)


# -- DBSCAN and spectral clustering --------------------------------------------------------------------------------

@builtin("dbscan")
def _dbscan(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("dbscan: expected dbscan(X, epsilon, minpts, 'Name', value, ...).")
    pos, opts = get_opts(args, 3, ["Distance", "P", "Cov", "Scale"], "dbscan")
    if len(pos) != 3:
        raise MatlabError("dbscan: expected dbscan(X, epsilon, minpts, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    eps, minpts = scalar(pos[1], "epsilon"), intval(pos[2], "minpts")
    if eps < 0 or minpts < 1:
        raise MatlabError("dbscan: epsilon must not be negative and minpts must be a positive integer.")
    if "distance" in opts and is_text(opts["distance"]) and text(opts["distance"]).lower() == "precomputed":
        D = X
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise MatlabError("dbscan: a precomputed distance matrix must be square.")
    else:
        metric = metric_name(opts["distance"], "dbscan") if "distance" in opts else "euclidean"
        D = dist_matrix(interp, X, X, metric, _nn_extra(opts, metric), "dbscan")
    n = D.shape[0]
    near = D <= eps
    core = near.sum(axis=1) >= minpts
    idx = np.full(n, -1.0)
    c = 0
    for i in range(n):
        if idx[i] != -1 or not core[i]:
            continue
        c += 1
        idx[i] = c
        stack = [i]
        while stack:
            j = stack.pop()
            for q in np.flatnonzero(near[j]):
                if idx[q] == -1:
                    idx[q] = c
                    if core[q]:
                        stack.append(q)
    return outs(nargout, idx.reshape(-1, 1), core.reshape(-1, 1))


@builtin("spectralcluster")
def _spectralcluster(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("spectralcluster: expected spectralcluster(X, k, 'Name', value, ...).")
    known = ["Distance", "P", "Cov", "Scale", "SimilarityGraph", "NumNeighbors", "KNNGraphType", "Radius",
             "KernelScale", "LaplacianNormalization", "ClusterMethod"]
    pos, opts = get_opts(args, 2, known, "spectralcluster")
    X = np.asarray(fnum(pos[0]), dtype=float)
    k = intval(pos[1], "k")
    n = X.shape[0]
    pre = "distance" in opts and is_text(opts["distance"]) and text(opts["distance"]).lower() == "precomputed"
    if pre:
        D = X
    else:
        metric = metric_name(opts["distance"], "spectralcluster") if "distance" in opts else "euclidean"
        D = dist_matrix(interp, X, X, metric, _nn_extra(opts, metric), "spectralcluster")
    graph = choice(opts["similaritygraph"], ["knn", "epsilon"], "SimilarityGraph") if "similaritygraph" in opts \
        else "knn"
    sigma = 1.0
    if "kernelscale" in opts:
        v = opts["kernelscale"]
        sigma = float(np.median(D[D > 0])) if is_text(v) else scalar(v, "KernelScale")
    S = np.exp(-(D / sigma) ** 2)
    if graph == "knn":
        nn = intval(opts["numneighbors"],
                "NumNeighbors") if "numneighbors" in opts else max(int(np.floor(np.log(n))), 1)
        kind = choice(opts["knngraphtype"], ["complete", "mutual"], "KNNGraphType") if "knngraphtype" in opts \
            else "complete"
        order = np.argsort(D, axis=1, kind="stable")[:, 1:nn + 1]
        A = np.zeros((n, n), dtype=bool)
        A[np.repeat(np.arange(n), order.shape[1]), order.ravel()] = True
        A = (A | A.T) if kind == "complete" else (A & A.T)
    else:
        if "radius" not in opts:
            raise MatlabError("spectralcluster: an 'epsilon' graph needs 'Radius'.")
        A = D <= scalar(opts["radius"], "Radius")
    np.fill_diagonal(A, False)
    W = np.where(A, S, 0.0)
    deg = W.sum(axis=1)
    norm = choice(opts["laplaciannormalization"], ["none", "randomwalk", "symmetric"], "LaplacianNormalization") \
        if "laplaciannormalization" in opts else "randomwalk"
    L = np.diag(deg) - W
    import scipy.linalg as sla
    dsafe = np.where(deg > 0, deg, 1.0)
    if norm == "none":
        ev, V = np.linalg.eigh(L)
    elif norm == "symmetric":
        ds = 1 / np.sqrt(dsafe)
        ev, V = np.linalg.eigh(ds[:, None] * L * ds[None, :])
    else:
        ev, V = sla.eigh(L, np.diag(dsafe))
    V, ev = V[:, :k], ev[:k]
    if norm == "symmetric":
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-300)
    method = choice(opts["clustermethod"], ["kmeans", "kmedoids"], "ClusterMethod") if "clustermethod" in opts \
        else "kmeans"
    rng = rng_of(interp)
    if method == "kmeans":
        starts = [_km_seed(rng, V, k, "sqeuclidean", "plus") for _ in range(5)]
        idx = kmeans_core(rng, V, k, "sqeuclidean", starts, 100, "singleton")[0]
    else:
        Dv = dist_matrix(interp, V, V, "squaredeuclidean", [], "spectralcluster")
        med, _ = _pam(rng, Dv, k, rng.choice(n, k, replace=False))
        idx = np.argmin(Dv[:, med], axis=1)
    return outs(nargout, (idx + 1.0).reshape(-1, 1), V, ev.reshape(-1, 1))


# -- Gaussian mixtures ---------------------------------------------------------------------------------------------

class GMDist(StatObject):
    matlab_class = "gmdistribution"

    def __init__(self, mu: np.ndarray, sigmas: list, props: np.ndarray, diagonal: bool, shared: bool):
        super().__init__()
        self.mu, self.sigmas, self.props = mu, sigmas, props   # sigmas: one full d-by-d matrix per component
        self.diagonal, self.shared = diagonal, shared
        self.info: dict = {}

    @property
    def k(self) -> int:
        return self.mu.shape[0]

    def log_comp(self, X: np.ndarray) -> np.ndarray:
        """log(p_j N(x | mu_j, Sigma_j)) for every point and component."""
        n, d = X.shape
        out = np.empty((n, self.k))
        for j in range(self.k):
            try:
                L = np.linalg.cholesky(self.sigmas[j])
            except np.linalg.LinAlgError:
                raise MatlabError("The covariance matrix of a mixture component is not positive definite.") from None
            z = np.linalg.solve(L, (X - self.mu[j]).T)
            with np.errstate(divide="ignore"):
                out[:, j] = (np.log(self.props[j]) - 0.5 * np.sum(z ** 2, axis=0) - np.sum(np.log(np.diag(L)))
                             - 0.5 * d * np.log(2 * np.pi))
        return out

    def _sigma_value(self) -> np.ndarray:
        d = self.mu.shape[1]
        if self.diagonal:
            S = np.stack([np.diag(s).reshape(1, d) for s in self.sigmas], axis=2)
        else:
            S = np.stack(self.sigmas, axis=2)
        return S[:, :, 0] if self.shared else S

    def nparams(self) -> int:
        d = self.mu.shape[1]
        per = d if self.diagonal else d * (d + 1) // 2
        return self.k - 1 + self.k * d + per * (1 if self.shared else self.k)

    def property_names(self) -> list[str]:
        return ["NumVariables", "DistributionName", "NumComponents", "ComponentProportion", "SharedCovariance",
                "NumIterations", "RegularizationValue", "NegativeLogLikelihood", "CovarianceType", "mu", "Sigma",
                "AIC", "BIC", "Converged", "ProbabilityTolerance"]

    def get_field(self, interp, name: str) -> Any:
        if name == "NumVariables":
            return float(self.mu.shape[1])
        if name == "DistributionName":
            return "gaussian mixture distribution"
        if name == "NumComponents":
            return float(self.k)
        if name in ("ComponentProportion", "PComponents"):
            return self.props.reshape(1, -1)
        if name in ("SharedCovariance", "SharedCov"):
            return bool(self.shared)
        if name in ("CovarianceType", "CovType"):
            return "diagonal" if self.diagonal else "full"
        if name == "mu":
            return self.mu.copy()
        if name == "Sigma":
            return self._sigma_value()
        if name in ("NumIterations", "RegularizationValue", "NegativeLogLikelihood", "AIC", "BIC", "Converged",
                    "ProbabilityTolerance"):
            v = self.info.get(name)
            return np.zeros((0, 0)) if v is None else v
        if name == "NlogL":
            v = self.info.get("NegativeLogLikelihood")
            return np.zeros((0, 0)) if v is None else v
        return super().get_field(interp, name)

    def _x(self, args: list, fname: str) -> np.ndarray:
        if len(args) != 1:
            raise MatlabError(f"{fname}: expected {fname}(gm, X).")
        X = np.asarray(fnum(args[0]), dtype=float)
        if X.ndim != 2 or X.shape[1] != self.mu.shape[1]:
            raise MatlabError(f"{fname}: X must have {self.mu.shape[1]} column(s).")
        return X

    def m_pdf(self, interp, args, nargout):
        from scipy.special import logsumexp
        return np.exp(logsumexp(self.log_comp(self._x(args, "pdf")), axis=1)).reshape(-1, 1)

    def m_cdf(self, interp, args, nargout):
        import scipy.stats as st
        X = self._x(args, "cdf")
        out = np.zeros(X.shape[0])
        for j in range(self.k):
            dist = st.multivariate_normal(mean=self.mu[j], cov=self.sigmas[j], allow_singular=True)
            out += self.props[j] * np.atleast_1d(dist.cdf(X))
        return out.reshape(-1, 1)

    def m_posterior(self, interp, args, nargout):
        from scipy.special import logsumexp
        lc = self.log_comp(self._x(args, "posterior"))
        tot = logsumexp(lc, axis=1)
        return outs(nargout, np.exp(lc - tot[:, None]), -float(tot.sum()))

    def m_cluster(self, interp, args, nargout):
        from scipy.special import logsumexp
        X = self._x(args, "cluster")
        lc = self.log_comp(X)
        tot = logsumexp(lc, axis=1)
        idx = np.argmax(lc, axis=1) + 1.0
        return outs(nargout, idx.reshape(-1, 1), -float(tot.sum()), np.exp(lc - tot[:, None]), tot.reshape(-1, 1),
                    self._mahal(X))

    def _mahal(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], self.k))
        for j in range(self.k):
            L = np.linalg.cholesky(self.sigmas[j])
            out[:, j] = np.sum(np.linalg.solve(L, (X - self.mu[j]).T) ** 2, axis=0)
        return out

    def m_mahal(self, interp, args, nargout):
        return self._mahal(self._x(args, "mahal"))

    def m_random(self, interp, args, nargout):
        if len(args) > 1:
            raise MatlabError("random: expected random(gm) or random(gm, n).")
        n = intval(args[0], "n") if args else 1
        rng = rng_of(interp)
        comp = rng.choice(self.k, size=n, p=self.props / self.props.sum())
        d = self.mu.shape[1]
        Y = np.empty((n, d))
        for j in range(self.k):
            sel = comp == j
            if sel.any():
                Y[sel] = self.mu[j] + rng.standard_normal((int(sel.sum()), d)) @ np.linalg.cholesky(self.sigmas[j]).T
        return outs(nargout, Y, (comp + 1.0).reshape(-1, 1))

    def display_lines(self) -> list[str]:
        d = self.mu.shape[1]
        lines = ["", f"Gaussian mixture distribution with {self.k} components in {d} dimensions"]
        for j in range(self.k):
            lines.append(f"Component {j + 1}:")
            lines.append(f"Mixing proportion: {self.props[j]:.6f}")
            lines.append("Mean:" + "".join(f"{v:10.4f}" for v in self.mu[j]))
            lines.append("")
        return lines

    def format_display(self, name: str) -> str:
        return f"{name} = \n" + "\n".join(self.display_lines()) + "\n"


def _gm_sigmas(S: np.ndarray, k: int, d: int, fname: str) -> tuple[list, bool, bool]:
    if S.ndim == 2:
        shared = True
        if S.shape == (d, d) and not (d == 1 and k > 1 and False):
            return [S.copy() for _ in range(k)], False, shared
        if S.shape == (1, d):
            return [np.diag(S.flatten()) for _ in range(k)], True, shared
    elif S.ndim == 3 and S.shape[2] == k:
        if S.shape[:2] == (d, d):
            return [S[:, :, j].copy() for j in range(k)], False, False
        if S.shape[:2] == (1, d):
            return [np.diag(S[0, :, j]) for j in range(k)], True, False
    raise MatlabError(f"{fname}: Sigma must be d-by-d, d-by-d-by-k, 1-by-d or 1-by-d-by-k.")


@builtin("gmdistribution")
def _gmdistribution(interp, args, nargout):
    if not 2 <= len(args) <= 3:
        raise MatlabError("gmdistribution: expected gmdistribution(mu, sigma) or gmdistribution(mu, sigma, p).")
    mu = np.asarray(fnum(args[0]), dtype=float)
    k, d = mu.shape
    S = np.asarray(fnum(args[1]), dtype=float)
    sigmas, diagonal, shared = _gm_sigmas(S, k, d, "gmdistribution")
    p = vec(args[2]) if given(args, 2) else np.full(k, 1.0 / k)
    if p.size != k or np.any(p < 0) or p.sum() <= 0:
        raise MatlabError("gmdistribution: p must hold one non-negative mixing proportion per component.")
    return GMDist(mu, sigmas, p / p.sum(), diagonal, shared)


def _em(X: np.ndarray, gm: GMDist, max_iter: int, tol: float, reg: float, ptol: float):
    from scipy.special import logsumexp
    n, d = X.shape
    k = gm.k
    prev = -np.inf
    converged = False
    it = 0
    ll = -np.inf
    for it in range(1, max_iter + 1):
        lc = gm.log_comp(X)
        tot = logsumexp(lc, axis=1)
        ll = float(tot.sum())
        R = np.exp(lc - tot[:, None])
        if ptol > 0:
            R[R < ptol] = 0
            R = R / R.sum(axis=1, keepdims=True)
        if abs(ll - prev) < tol * abs(ll) and it > 1:
            converged = True
            break
        prev = ll
        nk = R.sum(axis=0)
        if np.any(nk <= 0):
            raise MatlabError("fitgmdist: a component lost all of its points; try fewer components or regularization.")
        gm.props = nk / n
        gm.mu = (R.T @ X) / nk[:, None]
        covs = []
        for j in range(k):
            Xc = X - gm.mu[j]
            C = (Xc * R[:, j:j + 1]).T @ Xc / nk[j]
            covs.append(C)
        if gm.shared:
            pooled = sum(c * nk[j] for j, c in enumerate(covs)) / n
            covs = [pooled.copy() for _ in range(k)]
        for j in range(k):
            C = np.diag(np.diag(covs[j])) if gm.diagonal else (covs[j] + covs[j].T) / 2
            C = C + reg * np.eye(d)
            if np.linalg.cond(C) > 1e15 or np.any(np.diag(C) <= 0):
                raise MatlabError(f"fitgmdist: ill-conditioned covariance created at iteration {it}. Try "
                                  "'RegularizationValue', fewer components or 'CovarianceType', 'diagonal'.")
            covs[j] = C
        gm.sigmas = covs
    return ll, it, converged


_GM_OPTS = ["CovarianceType", "CovType", "SharedCovariance", "SharedCov", "RegularizationValue", "Regularize",
            "Replicates", "Start", "Options", "ProbabilityTolerance"]


@builtin("fitgmdist", "gmdistribution.fit")
def _fitgmdist(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("fitgmdist: expected fitgmdist(X, k, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, _GM_OPTS, "fitgmdist")
    if len(pos) != 2:
        raise MatlabError("fitgmdist: expected fitgmdist(X, k, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError("fitgmdist: X must be a matrix with one observation per row.")
    X = X[~np.isnan(X).any(axis=1)]
    n, d = X.shape
    k = intval(pos[1], "k")
    if k < 1 or n <= d or n < k:
        raise MatlabError("fitgmdist: X needs more rows than columns and at least k rows.")
    ct = opts.get("covariancetype", opts.get("covtype"))
    diagonal = ct is not None and choice(ct, ["full", "diagonal"], "CovarianceType") == "diagonal"
    sh = opts.get("sharedcovariance", opts.get("sharedcov"))
    shared = as_bool(sh, "SharedCovariance") if sh is not None else False
    rv = opts.get("regularizationvalue", opts.get("regularize"))
    reg = scalar(rv, "RegularizationValue") if rv is not None else 0.0
    ptol = scalar(opts["probabilitytolerance"], "ProbabilityTolerance") if "probabilitytolerance" in opts else 1e-8
    max_iter, tol = 100, 1e-6
    o = opts.get("options")
    if isinstance(o, Struct):
        for nm in ("MaxIter", "TolFun"):
            if nm in o.fields and not is_empty_arg(o.get_field(0, nm)):
                if nm == "MaxIter":
                    max_iter = intval(o.get_field(0, nm), nm)
                else:
                    tol = scalar(o.get_field(0, nm), nm)
    elif o is not None and not is_empty_arg(o):
        raise MatlabError("fitgmdist: Options must be a structure created by statset.")
    reps = intval(opts["replicates"], "Replicates") if "replicates" in opts else 1
    rng = rng_of(interp)
    start = opts.get("start")
    var0 = np.diag(X.var(axis=0, ddof=1)) + reg * np.eye(d)

    def initial() -> GMDist:
        if isinstance(start, Struct):
            mu = np.asarray(fnum(start.get_field(0, "mu")), dtype=float)
            sig, dg, shd = _gm_sigmas(np.asarray(fnum(start.get_field(0, "Sigma")), dtype=float), k, d, "fitgmdist")
            p = vec(start.get_field(0, "ComponentProportion")) if "ComponentProportion" in start.fields \
                else np.full(k, 1.0 / k)
            g = GMDist(mu, sig, p / p.sum(), diagonal or dg, shared or shd)
            return g
        if start is not None and not is_text(start):
            lab = vec(start).astype(int) - 1
            if lab.size != n or lab.min() < 0 or lab.max() >= k:
                raise MatlabError("fitgmdist: a Start vector must assign every observation to one of k components.")
            mu = np.array([X[lab == j].mean(axis=0) for j in range(k)])
            sig = []
            for j in range(k):
                M = X[lab == j]
                C = np.cov(M, rowvar=False).reshape(d, d) if M.shape[0] > 1 else var0.copy()
                sig.append((np.diag(np.diag(C)) if diagonal else C) + reg * np.eye(d))
            if shared:
                pooled = sum(s * np.sum(lab == j) for j, s in enumerate(sig)) / n
                sig = [pooled.copy() for _ in range(k)]
            return GMDist(mu, sig, np.bincount(lab, minlength=k) / n, diagonal, shared)
        mode = choice(start, ["plus", "randSample"], "Start") if start is not None else "plus"
        mu = _km_seed(rng, X, k, "sqeuclidean", "plus" if mode == "plus" else "sample")
        return GMDist(mu.copy(), [var0.copy() for _ in range(k)], np.full(k, 1.0 / k), diagonal, shared)

    best = None
    last_error = None
    for _ in range(reps if not isinstance(start, Struct) and (start is None or is_text(start)) else 1):
        gm = initial()
        try:
            ll, it, conv = _em(X, gm, max_iter, tol, reg, ptol)
        except MatlabError as e:
            last_error = e
            continue
        if best is None or ll > best[0]:
            best = (ll, gm, it, conv)
    if best is None:
        raise last_error
    ll, gm, it, conv = best
    npar = gm.nparams()
    gm.info = {"NumIterations": float(it), "RegularizationValue": float(reg), "NegativeLogLikelihood": -ll,
               "AIC": -2 * ll + 2 * npar, "BIC": -2 * ll + npar * np.log(n), "Converged": bool(conv),
               "ProbabilityTolerance": float(ptol)}
    return gm


# -- silhouettes and cluster evaluation ------------------------------------------------------------------------------

def silhouette_values(D: np.ndarray, codes: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    ks = np.unique(codes[codes >= 0])
    s = np.full(n, np.nan)
    sums = np.column_stack([D[:, codes == c].sum(axis=1) for c in ks]) if ks.size else np.zeros((n, 0))
    counts = np.array([np.sum(codes == c) for c in ks], dtype=float)
    for i in range(n):
        if codes[i] < 0:
            continue
        own = int(np.searchsorted(ks, codes[i]))
        if counts[own] <= 1:
            s[i] = 1.0
            continue
        a = sums[i, own] / (counts[own] - 1)
        others = np.delete(sums[i] / counts, own)
        if others.size == 0:
            s[i] = np.nan
            continue
        b = others.min()
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return s


def _sil_dist(interp, X: np.ndarray, metric_arg: Any, extra: list, fname: str) -> np.ndarray:
    if metric_arg is not None and not is_text(metric_arg) and not isinstance(metric_arg, FunctionHandle):
        y = np.asarray(fnum(metric_arg), dtype=float)
        if not _is_dist_vector(y):
            raise MatlabError(f"{fname}: a numeric distance must be a vector as returned by pdist.")
        n = X.shape[0]
        Dm = np.zeros((n, n))
        Dm[np.triu_indices(n, 1)] = y.flatten()
        return Dm + Dm.T
    metric = metric_name(metric_arg, fname) if metric_arg is not None else "squaredeuclidean"
    return dist_matrix(interp, X, X, metric, extra, fname)


@builtin("silhouette")
def _silhouette(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("silhouette: expected silhouette(X, clust) or silhouette(X, clust, distance).")
    X = np.asarray(fnum(args[0]), dtype=float)
    codes, names, _ = group_codes(args[1])
    if codes.size != X.shape[0]:
        raise MatlabError("silhouette: clust must have one value per row of X.")
    rest = list(args[2:])
    if rest and is_text(rest[0]) and text(rest[0]).lower() == "distance":
        rest = rest[1:]
    D = _sil_dist(interp, X, rest[0] if rest else None, rest[1:], "silhouette")
    s = silhouette_values(D, codes)
    if nargout == 1:
        return s.reshape(-1, 1)
    order = np.lexsort((-np.nan_to_num(s, nan=-2.0), codes))
    order = order[codes[order] >= 0]
    builtin_call(interp, "barh", [np.arange(1.0, order.size + 1).reshape(1, -1), s[order].reshape(1, -1),
                                  np.array([[1.0]])], 0)
    builtin_call(interp, "xlabel", [char("Silhouette Value")], 0)
    builtin_call(interp, "ylabel", [char("Cluster")], 0)
    h = builtin_call(interp, "gcf", [], 1)[0]
    return outs(nargout, s.reshape(-1, 1), h) if nargout else None


def _criterion_value(name: str, X: np.ndarray, codes: np.ndarray, D: np.ndarray | None) -> float:
    ks = np.unique(codes[codes >= 0])
    k, n = ks.size, int(np.sum(codes >= 0))
    with np.errstate(all="ignore"):
        if name == "silhouette":
            return float(np.nanmean(silhouette_values(D, codes)))
        cents = np.array([X[codes == c].mean(axis=0) for c in ks])
        sizes = np.array([np.sum(codes == c) for c in ks], dtype=float)
        if name == "calinskiharabasz":
            grand = X[codes >= 0].mean(axis=0)
            ssb = float(np.sum(sizes * np.sum((cents - grand) ** 2, axis=1)))
            ssw = float(sum(np.sum((X[codes == c] - cents[i]) ** 2) for i, c in enumerate(ks)))
            return (ssb / (k - 1)) / (ssw / (n - k)) if k > 1 and n > k and ssw > 0 else np.nan
        spread = np.array([np.mean(np.sqrt(np.sum((X[codes == c] - cents[i]) ** 2, axis=1))) for i, c in enumerate(ks)])
        if k < 2:
            return np.nan
        dc = np.sqrt(np.sum((cents[:, None, :] - cents[None]) ** 2, axis=2))
        R = (spread[:, None] + spread[None, :]) / np.where(dc > 0, dc, np.inf)
        np.fill_diagonal(R, -np.inf)
        return float(np.mean(R.max(axis=1)))


class ClusterEvaluation(StatObject):
    def __init__(self, cname: str, props: dict):
        super().__init__()
        self.matlab_class = f"{cname}Evaluation"
        self.p = props

    def m_plot(self, interp, args, nargout):
        ks = np.asarray(self.p["InspectedK"], dtype=float)
        v = np.asarray(self.p["CriterionValues"], dtype=float)
        h = builtin_call(interp, "plot", [ks.reshape(1, -1), v.reshape(1, -1), char("-o")], 1)
        builtin_call(interp, "xlabel", [char("Number of Clusters")], 0)
        builtin_call(interp, "ylabel", [char(f"{self.p['CriterionName']} Values")], 0)
        return h[0] if nargout else None


@builtin("evalclusters")
def _evalclusters(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("evalclusters: expected evalclusters(X, clust, criterion, 'KList', ks).")
    known = ["KList", "Distance", "ClusterPriors", "B", "ReferenceDistribution", "SearchMethod"]
    pos, opts = get_opts(args, 3, known, "evalclusters")
    if len(pos) != 3:
        raise MatlabError("evalclusters: expected evalclusters(X, clust, criterion, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    cname = choice(pos[2], ["CalinskiHarabasz", "DaviesBouldin", "silhouette", "gap"], "The criterion")
    if "clusterpriors" in opts:
        raise MatlabError("evalclusters: the option 'ClusterPriors' is not supported.")
    keep = ~np.isnan(X).any(axis=1)
    Xk = X[keep]
    clust = pos[1]
    rng = rng_of(interp)
    metric = metric_name(opts["distance"], "evalclusters") if "distance" in opts else "squaredeuclidean"
    D = dist_matrix(interp, Xk, Xk, metric, [], "evalclusters") if cname == "silhouette" else None

    def assign(data: np.ndarray, k: int) -> np.ndarray:
        if isinstance(clust, FunctionHandle):
            return vec(call(interp, clust, [data, np.array([[float(k)]])], 1)[0]).astype(int) - 1
        kind = choice(clust, ["kmeans", "linkage", "gmdistribution"], "The clustering algorithm")
        if kind == "kmeans":
            starts = [_km_seed(rng, data, k, "sqeuclidean", "plus") for _ in range(5)]
            return kmeans_core(rng, data, k, "sqeuclidean", starts, 100, "singleton")[0]
        if kind == "linkage":
            Z = linkage_core(interp, data, "ward", None)
            return cluster_tree(Z, {"maxclust": np.array([[float(k)]])}, "evalclusters")[:, 0].astype(int) - 1
        gm = _fitgmdist(interp, [data, np.array([[float(k)]]), char("Replicates"), np.array([[5.0]]),
                                 char("RegularizationValue"), np.array([[1e-6]])], 1)
        return np.argmax(gm.log_comp(data), axis=1)

    if not is_text(clust) and not isinstance(clust, FunctionHandle):
        sols = np.asarray(fnum(clust), dtype=float)
        if sols.ndim != 2 or sols.shape[0] != X.shape[0]:
            raise MatlabError("evalclusters: a matrix of clustering solutions needs one row per observation.")
        if cname == "gap":
            raise MatlabError("evalclusters: the gap criterion needs a clustering algorithm, not fixed solutions.")
        solutions = [group_codes(sols[keep, j].reshape(-1, 1))[0] for j in range(sols.shape[1])]
        ks = np.array([float(np.unique(s[s >= 0]).size) for s in solutions])
    else:
        if "klist" not in opts:
            raise MatlabError("evalclusters: 'KList' is needed when clust is an algorithm.")
        ks = vec(opts["klist"])
        solutions = [assign(Xk, int(k)) for k in ks]
    key = cname.lower()
    props: dict = {"NumObservations": float(X.shape[0]), "InspectedK": ks.reshape(1, -1)}
    if cname == "gap":
        B = intval(opts["b"], "B") if "b" in opts else 100
        ref = choice(opts["referencedistribution"], ["PCA", "uniform"], "ReferenceDistribution") \
            if "referencedistribution" in opts else "pca"

        def logw(data, codes):
            tot = 0.0
            for c in np.unique(codes):
                M = data[codes == c]
                tot += np.sum((M - M.mean(axis=0)) ** 2)
            return np.log(max(tot, 1e-300))

        mu = Xk.mean(axis=0)
        if ref == "pca":
            _, _, Vt = np.linalg.svd(Xk - mu, full_matrices=False)
            base = (Xk - mu) @ Vt.T
        else:
            Vt, base = np.eye(Xk.shape[1]), Xk - mu
        lo, hi = base.min(axis=0), base.max(axis=0)
        gap, se, lw, elw = [], [], [], []
        for k, codes in zip(ks, solutions):
            obs = logw(Xk, codes)
            sims = []
            for _ in range(B):
                R = (lo + (hi - lo) * rng.random(base.shape)) @ Vt + mu
                sims.append(logw(R, assign(R, int(k))))
            sims = np.array(sims)
            gap.append(sims.mean() - obs)
            se.append(sims.std(ddof=0) * np.sqrt(1 + 1.0 / B))
            lw.append(obs)
            elw.append(sims.mean())
        vals = np.array(gap)
        se = np.array(se)
        method = choice(opts["searchmethod"], ["globalMaxSE", "firstMaxSE"], "SearchMethod") \
            if "searchmethod" in opts else "globalmaxse"
        if method == "globalmaxse":
            g = int(np.argmax(vals))
            cand = np.flatnonzero(vals >= vals[g] - se[g])
            best = int(cand[0])
        else:
            best = len(ks) - 1
            for i in range(len(ks) - 1):
                if vals[i] >= vals[i + 1] - se[i + 1]:
                    best = i
                    break
        props.update({"CriterionName": "Gap", "B": float(B), "SE": se.reshape(1, -1),
                      "LogW": np.array(lw).reshape(1, -1), "ExpectedLogW": np.array(elw).reshape(1, -1),
                      "ReferenceDistribution": "PCA" if ref == "pca" else "uniform"})
    else:
        if "b" in opts or "referencedistribution" in opts or "searchmethod" in opts:
            raise MatlabError("evalclusters: B, ReferenceDistribution and SearchMethod apply to the gap criterion.")
        vals = np.array([_criterion_value(key, Xk, codes, D) for codes in solutions])
        with np.errstate(all="ignore"):
            best = int(np.nanargmin(vals)) if key == "daviesbouldin" else int(np.nanargmax(vals))
        props["CriterionName"] = {"calinskiharabasz": "CalinskiHarabasz", "daviesbouldin": "DaviesBouldin",
                                  "silhouette": "Silhouette"}[key]
    opt_y = np.full((X.shape[0], 1), np.nan)
    opt_y[keep, 0] = solutions[best] + 1.0
    props.update({"CriterionValues": vals.reshape(1, -1), "OptimalK": float(ks[best]), "OptimalY": opt_y,
                  "Missing": (~keep).reshape(-1, 1),
                  "ClusteringFunction": text(clust) if is_text(clust) else ("" if not isinstance(clust, FunctionHandle)
                                                                            else repr(clust)),
                  "X": X})
    return ClusterEvaluation({"calinskiharabasz": "CalinskiHarabasz", "daviesbouldin": "DaviesBouldin",
                              "silhouette": "Silhouette", "gap": "Gap"}[key], props)
