"""graph and digraph objects, graph algorithms, and the sparse structure functions built on them.

A :class:`GraphObject` holds the node count, optional node names, an edge list kept in the
canonical order (by source node, then by target node) and optional weights. Extra columns of the
``Nodes`` and ``Edges`` tables are carried along. The algorithms are written directly on the edge
list; nothing here needs SciPy at import time.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, StringArray, char, is_char, to_int, to_str
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, fnum, is_text, text


# -- small helpers ------------------------------------------------------------------------------

def _is_empty_arg(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.size == 0


def _is_names(v: Any) -> bool:
    """True for node names: char row, string, string array or cell array of text."""
    if isinstance(v, (MString, StringArray)):
        return True
    if isinstance(v, Cell):
        return all(is_text(x) for x in v.values_f())
    return is_char(v) and v.size > 0


def _name_list(v: Any) -> list[str]:
    if isinstance(v, MString):
        return [str(v)]
    if isinstance(v, StringArray):
        return [str(x) for x in v.values_f()]
    if isinstance(v, Cell):
        return [to_str(x) for x in v.values_f()]
    if is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return ["".join(row).rstrip() for row in v]
        return [to_str(v)]
    raise MatlabError("Node names must be text.")


def _names_cell(names: list[str], column: bool = True) -> Cell:
    return Cell.from_list([char(s) for s in names], column=column)


def _col(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1, 1)


def _row(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(1, -1)


def _table_cls():
    from .table import TableObject
    return TableObject


_NODE_TABLE: list = []


def _sized_table(names: list[str], columns: list, nrows: int):
    """A table; one without variables still reports ``nrows`` rows (the Nodes table of a plain graph)."""
    base = _table_cls()
    if columns:
        return base(names, columns)
    if not _NODE_TABLE:
        class NodeTable(base):  # type: ignore[misc, valid-type]
            forced_rows = 0

            @property
            def nrows(self) -> int:
                return base.nrows.fget(self) if self.columns else self.forced_rows

            def copy(self):
                t = NodeTable(list(self.names), list(self.columns), self.description, self.row_names)
                t.forced_rows = self.forced_rows
                return t

        _NODE_TABLE.append(NodeTable)
    t = _NODE_TABLE[0]([], [])
    t.forced_rows = nrows
    return t


def _take_rows(col: Any, idx: np.ndarray) -> Any:
    from .table import _rows
    return _rows(col, np.asarray(idx, dtype=int))


def _col_height(col: Any) -> int:
    from .table import _height
    return _height(col)


def _cat_cols(a: Any, b: Any) -> Any:
    from ..ops import concat
    return concat([a, b], 0)


def _filler(col: Any, n: int) -> Any:
    """n rows of the missing value that matches a table column."""
    if isinstance(col, Cell):
        text_cells = all(is_char(x) for x in col.values_f())
        d = np.empty((n, col.data.shape[1]), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.zeros((0, 0), dtype="<U1") if text_cells else np.zeros((0, 0))
        return Cell(d)
    if isinstance(col, StringArray):
        d = np.empty((n, col.data.shape[1]), dtype=object)
        return StringArray(d)
    if isinstance(col, np.ndarray):
        if col.dtype.kind == "f":
            return np.full((n,) + col.shape[1:], np.nan, dtype=col.dtype)
        if col.dtype.kind == "U":
            return np.full((n,) + col.shape[1:], " ", dtype=col.dtype)
        return np.zeros((n,) + col.shape[1:], dtype=col.dtype)
    raise MatlabError("Unsupported table variable type in a graph table.")


# -- the graph object ----------------------------------------------------------------------------------

class GraphObject(MObject):
    """An undirected (``graph``) or directed (``digraph``) graph."""

    def __init__(self, directed: bool, n: int, src, dst, weights=None, names: list[str] | None = None,
                 edge_vars: dict | None = None, node_vars: dict | None = None, presorted: bool = False):
        self.directed = directed
        self.n = int(n)
        src = np.asarray(src, dtype=np.int64).reshape(-1)
        dst = np.asarray(dst, dtype=np.int64).reshape(-1)
        if not directed and src.size:
            lo, hi = np.minimum(src, dst), np.maximum(src, dst)
            src, dst = lo, hi
        w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        ev = dict(edge_vars or {})
        if not presorted and src.size:
            order = np.lexsort((dst, src))  # stable: parallel edges keep their relative order
            if not np.array_equal(order, np.arange(src.size)):
                src, dst = src[order], dst[order]
                if w is not None:
                    w = w[order]
                ev = {k: _take_rows(v, order) for k, v in ev.items()}
        self.src, self.dst, self.w = src, dst, w
        self.names = list(names) if names is not None else None
        self.edge_vars = ev
        self.node_vars = dict(node_vars or {})
        self._adj: list | None = None
        self._radj: list | None = None

    # -- identity ------------------------------------------------------------------------------
    def class_name(self) -> str:
        return "digraph" if self.directed else "graph"

    def property_names(self) -> list[str]:
        return ["Edges", "Nodes"]

    def isa_class(self, name: str) -> bool:
        return name == self.class_name()

    def matlab_shape(self) -> tuple:
        return (1, 1)

    @property
    def m(self) -> int:
        return int(self.src.size)

    def weights(self, default: float = 1.0) -> np.ndarray:
        return self.w if self.w is not None else np.full(self.m, default)

    def clone(self, **changes) -> "GraphObject":
        args = dict(directed=self.directed, n=self.n, src=self.src, dst=self.dst, weights=self.w,
                    names=self.names, edge_vars=self.edge_vars, node_vars=self.node_vars, presorted=True)
        args.update(changes)
        return GraphObject(**args)

    def object_equals(self, other: Any) -> bool:
        if not isinstance(other, GraphObject) or other.directed != self.directed or other.n != self.n:
            return False
        if not (np.array_equal(self.src, other.src) and np.array_equal(self.dst, other.dst)):
            return False
        if (self.w is None) != (other.w is None) or (self.w is not None and not np.array_equal(self.w, other.w)):
            return False
        from ..values import values_equal
        if self.names != other.names or list(self.edge_vars) != list(other.edge_vars) \
                or list(self.node_vars) != list(other.node_vars):
            return False
        return all(values_equal(v, other.edge_vars[k]) for k, v in self.edge_vars.items()) and \
            all(values_equal(v, other.node_vars[k]) for k, v in self.node_vars.items())

    # -- adjacency lists -------------------------------------------------------------------------
    def out_lists(self) -> list:
        """Per node: list of (neighbor, edge index) following outgoing edges (all edges if undirected)."""
        if self._adj is None:
            adj: list = [[] for _ in range(self.n)]
            for e in range(self.m):
                s, t = int(self.src[e]), int(self.dst[e])
                adj[s].append((t, e))
                if not self.directed and s != t:
                    adj[t].append((s, e))
            if not self.directed:
                for lst in adj:
                    lst.sort()
            self._adj = adj
        return self._adj

    def in_lists(self) -> list:
        if not self.directed:
            return self.out_lists()
        if self._radj is None:
            radj: list = [[] for _ in range(self.n)]
            for e in range(self.m):
                radj[int(self.dst[e])].append((int(self.src[e]), e))
            for lst in radj:
                lst.sort()
            self._radj = radj
        return self._radj

    # -- node references ---------------------------------------------------------------------------
    def node_ids(self, v: Any, what: str = "node") -> tuple[np.ndarray, bool]:
        """0-based node indices for numeric IDs or names; the flag says names were given."""
        if _is_names(v):
            if self.names is None:
                raise MatlabError("The graph has no node names; refer to nodes by their numeric IDs.")
            out = []
            for s in _name_list(v):
                if s not in self._name_index():
                    raise MatlabError(f"Node '{s}' is not in the graph.")
                out.append(self._name_index()[s])
            return np.array(out, dtype=np.int64), True
        a = np.real(fnum(v)).reshape(-1, order="F") if not _is_empty_arg(v) else np.zeros(0)
        if a.size and (np.any(a != np.round(a)) or np.any(a < 1) or np.any(a > self.n)):
            raise MatlabError(f"Node IDs must be positive integers no larger than the number of nodes "
                              f"({self.n}).")
        return a.astype(np.int64) - 1, False

    def _name_index(self) -> dict:
        return {s: i for i, s in enumerate(self.names or [])}

    def one_node(self, v: Any, what: str = "node") -> tuple[int, bool]:
        ids, named = self.node_ids(v, what)
        if ids.size != 1:
            raise MatlabError(f"Expected a single {what}.")
        return int(ids[0]), named

    def node_out(self, ids, named: bool, column: bool = True) -> Any:
        """Node indices as MATLAB values: names (cell) when the caller used names."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if named and self.names is not None:
            return _names_cell([self.names[i] for i in ids], column=column)
        return _col(ids + 1.0) if column else _row(ids + 1.0)

    # -- tables ---------------------------------------------------------------------------------
    def edges_table(self):
        if self.names is not None:
            data = np.empty((self.m, 2), dtype=object)
            for e in range(self.m):
                data[e, 0] = char(self.names[int(self.src[e])])
                data[e, 1] = char(self.names[int(self.dst[e])])
            end = Cell(data)
        else:
            end = np.column_stack([self.src + 1.0, self.dst + 1.0]).reshape(self.m, 2)
        names, cols = ["EndNodes"], [end]
        if self.w is not None:
            names.append("Weight")
            cols.append(self.w.reshape(-1, 1).copy())
        for k, v in self.edge_vars.items():
            names.append(k)
            cols.append(v)
        return _table_cls()(names, cols)

    def nodes_table(self):
        names, cols = [], []
        if self.names is not None:
            names.append("Name")
            cols.append(_names_cell(self.names))
        for k, v in self.node_vars.items():
            names.append(k)
            cols.append(v)
        return _sized_table(names, cols, self.n)

    # -- protocol -----------------------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        return name in _METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        fn = _METHODS.get(name)
        if fn is None:
            return super().call_method(interp, name, args, nargout)
        res = fn(interp, list(args), nargout)
        if res is None:
            return []
        from ..values import to_matlab
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    def get_field(self, interp, name: str) -> Any:
        if name == "Edges":
            return self.edges_table()
        if name == "Nodes":
            return self.nodes_table()
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def set_field(self, interp, name: str, value: Any) -> "GraphObject":
        table_cls = _table_cls()
        if name == "Edges":
            if not isinstance(value, table_cls):
                raise MatlabError("Edges must be a table.")
            if "EndNodes" not in value.names:
                raise MatlabError("The Edges table must keep its EndNodes variable.")
            if value.nrows != self.m:
                raise MatlabError("The number of edges cannot be changed by assigning to Edges; "
                                  "use addedge or rmedge.")
            src, dst = _end_nodes(value.columns[value.names.index("EndNodes")], self)
            ok = np.array_equal(src, self.src) and np.array_equal(dst, self.dst) if self.directed else \
                np.array_equal(np.minimum(src, dst), self.src) and np.array_equal(np.maximum(src, dst), self.dst)
            if not ok:
                raise MatlabError("EndNodes cannot be modified directly; use addedge, rmedge or reordernodes.")
            w = None
            ev = {}
            for k, c in zip(value.names, value.columns):
                if k == "EndNodes":
                    continue
                if k == "Weight":
                    wa = fnum(c)
                    if wa.dtype.kind == "c" or wa.shape != (self.m, 1):
                        raise MatlabError("Edge weights must be a real numeric column vector.")
                    w = wa.astype(float).reshape(-1)
                else:
                    ev[k] = c
            return self.clone(weights=w, edge_vars=ev)
        if name == "Nodes":
            if not isinstance(value, table_cls):
                raise MatlabError("Nodes must be a table.")
            if value.names and value.nrows != self.n:
                raise MatlabError("The number of nodes cannot be changed by assigning to Nodes; "
                                  "use addnode or rmnode.")
            names = None
            nv = {}
            for k, c in zip(value.names, value.columns):
                if k == "Name":
                    names = _check_names(_name_list(c), self.n)
                else:
                    nv[k] = c
            return self.clone(names=names, node_vars=nv)
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def display_lines(self) -> list[str]:
        et, nt = self.edges_table(), self.nodes_table()
        return [f"  {self.class_name()} with properties:", "",
                f"    Edges: [{self.m}×{len(et.names)} table]",
                f"    Nodes: [{self.n}×{len(nt.names)} table]"]


def _check_names(names: list[str], n: int) -> list[str]:
    if len(names) != n:
        raise MatlabError(f"The number of node names ({len(names)}) must match the number of nodes ({n}).")
    if len(set(names)) != len(names):
        raise MatlabError("Node names must be unique.")
    if any(s == "" for s in names):
        raise MatlabError("Node names must be nonempty.")
    return names


def _end_nodes(col: Any, g: "GraphObject | None", names_out: list | None = None):
    """Source and target indices (0-based) from an EndNodes table column."""
    if isinstance(col, (Cell, StringArray)):
        data = col.data
        if data.ndim != 2 or data.shape[1] != 2:
            raise MatlabError("EndNodes must have two columns.")
        s = [to_str(x) for x in data[:, 0]]
        t = [to_str(x) for x in data[:, 1]]
        if g is not None:
            idx = g._name_index()
            try:
                return (np.array([idx[x] for x in s], dtype=np.int64),
                        np.array([idx[x] for x in t], dtype=np.int64))
            except KeyError as e:
                raise MatlabError(f"Node {e} is not in the graph.")
        order: dict[str, int] = {}
        for x in s + t:
            order.setdefault(x, len(order))
        if names_out is not None:
            names_out.extend(order)
        return (np.array([order[x] for x in s], dtype=np.int64),
                np.array([order[x] for x in t], dtype=np.int64))
    a = np.real(fnum(col))
    if a.ndim != 2 or a.shape[1] != 2:
        raise MatlabError("EndNodes must have two columns.")
    if a.size and (np.any(a != np.round(a)) or np.any(a < 1)):
        raise MatlabError("Node IDs in EndNodes must be positive integers.")
    return a[:, 0].astype(np.int64) - 1, a[:, 1].astype(np.int64) - 1


# -- construction ----------------------------------------------------------------------------------------

def _split_table(t, key: str, what: str) -> tuple[Any, dict]:
    """Take the key variable out of a table and return it with the remaining variables."""
    rest = {}
    main = None
    for k, c in zip(t.names, t.columns):
        if k == key:
            main = c
        else:
            rest[k] = c
    return main, rest


def _construct(directed: bool, args: list) -> GraphObject:
    kind = "digraph" if directed else "graph"
    table_cls = _table_cls()
    args = list(args)
    omit_loops = False
    tri = None
    # trailing flags
    while args and is_text(args[-1]) and not isinstance(args[-1], Cell) \
            and text(args[-1]).lower() in ("omitselfloops", "upper", "lower"):
        flag = text(args.pop()).lower()
        if flag == "omitselfloops":
            omit_loops = True
        elif directed:
            raise MatlabError("digraph: the 'upper' and 'lower' options apply only to graph.")
        else:
            tri = flag
    if not args:
        return GraphObject(directed, 0, [], [])
    first = args[0]
    names: list[str] | None = None
    node_vars: dict = {}
    edge_vars: dict = {}
    weights = None
    num = None

    def node_spec(v):
        nonlocal names, node_vars, num
        if isinstance(v, table_cls):
            nm, rest = _split_table(v, "Name", "Nodes")
            node_vars = rest
            num = v.nrows
            if nm is not None:
                names = _name_list(nm)
        elif _is_names(v):
            names = _name_list(v)
            num = len(names)
        elif _is_empty_arg(v):
            return
        else:
            num = to_int(v, "the number of nodes")
            if num < 0:
                raise MatlabError(f"{kind}: the number of nodes must be nonnegative.")

    if isinstance(first, table_cls):
        # graph(EdgeTable) or graph(EdgeTable, NodeTable)
        if len(args) > 2:
            raise MatlabError(f"Too many input arguments for {kind}.")
        end, rest = _split_table(first, "EndNodes", "Edges")
        if end is None:
            raise MatlabError(f"{kind}: the edge table must have a variable named EndNodes.")
        if len(args) > 1:
            node_spec(args[1])
        found: list[str] = []
        if isinstance(end, (Cell, StringArray)) and names is not None:
            tmp = GraphObject(directed, len(names), [], [], names=names)
            src, dst = _end_nodes(end, tmp)
        else:
            src, dst = _end_nodes(end, None, found)
            if found:
                names = found
        if "Weight" in rest:
            weights = np.real(fnum(rest.pop("Weight"))).astype(float).reshape(-1)
        edge_vars = rest
    elif len(args) >= 2 and (_is_names(first) or (isinstance(args[1], np.ndarray) and not is_char(args[1]))):
        # graph(s, t, ...): both node lists are indices, or both are names
        s_v, t_v = args[0], args[1]
        rest_args = args[2:]
        if len(rest_args) > 2:
            raise MatlabError(f"Too many input arguments for {kind}.")
        if rest_args:
            wv = rest_args[0]
            if isinstance(wv, table_cls):
                _, erest = _split_table(wv, "EndNodes", "Edges")
                if "EndNodes" in wv.names:
                    raise MatlabError(f"{kind}: an edge table given with s and t must not contain EndNodes.")
                if "Weight" in erest:
                    weights = np.real(fnum(erest.pop("Weight"))).astype(float).reshape(-1)
                edge_vars = erest
            elif not _is_empty_arg(wv):
                wa = fnum(wv)
                if wa.dtype.kind == "c":
                    raise MatlabError(f"{kind}: edge weights must be real.")
                weights = wa.astype(float).reshape(-1, order="F")
        if len(rest_args) > 1:
            node_spec(rest_args[1])
        if _is_names(s_v) != _is_names(t_v):
            raise MatlabError(f"{kind}: s and t must both be node indices or both be node names.")
        if _is_names(s_v):
            s_names, t_names = _name_list(s_v), _name_list(t_v)
            if len(s_names) == 1 and len(t_names) > 1:
                s_names = s_names * len(t_names)
            if len(t_names) == 1 and len(s_names) > 1:
                t_names = t_names * len(s_names)
            if len(s_names) != len(t_names):
                raise MatlabError(f"{kind}: s and t must have the same number of elements.")
            if names is None:
                order: dict[str, int] = {}
                for x in s_names + t_names:
                    order.setdefault(x, len(order))
                names = list(order)
            idx = {x: i for i, x in enumerate(names)}
            try:
                src = np.array([idx[x] for x in s_names], dtype=np.int64)
                dst = np.array([idx[x] for x in t_names], dtype=np.int64)
            except KeyError as e:
                raise MatlabError(f"{kind}: node {e} is not in the list of node names.")
        else:
            sa = np.real(fnum(s_v)).reshape(-1, order="F")
            ta = np.real(fnum(t_v)).reshape(-1, order="F")
            if sa.size == 1 and ta.size > 1:
                sa = np.full(ta.size, sa[0])
            if ta.size == 1 and sa.size > 1:
                ta = np.full(sa.size, ta[0])
            if sa.size != ta.size:
                raise MatlabError(f"{kind}: s and t must have the same number of elements.")
            for a in (sa, ta):
                if a.size and (np.any(a != np.round(a)) or np.any(a < 1)):
                    raise MatlabError(f"{kind}: node indices must be positive integers.")
            src, dst = sa.astype(np.int64) - 1, ta.astype(np.int64) - 1
        if weights is not None:
            if weights.size == 1 and src.size != 1:
                weights = np.full(src.size, weights[0])
            if weights.size != src.size:
                raise MatlabError(f"{kind}: the number of weights must match the number of edges.")
        for v in edge_vars.values():
            if _col_height(v) != src.size:
                raise MatlabError(f"{kind}: the edge table must have one row per edge.")
    else:
        # graph(A, ...)
        if len(args) > 2:
            raise MatlabError(f"Too many input arguments for {kind}.")
        a = _adjacency_arg(first, kind)
        n = a.shape[0]
        if len(args) > 1:
            node_spec(args[1])
            if num is not None and num != n:
                raise MatlabError(f"{kind}: the number of node names must match the size of the adjacency matrix.")
        import scipy.sparse as sp
        logical = a.dtype == bool
        coo = sp.coo_matrix(a)
        r, c, v = coo.row.astype(np.int64), coo.col.astype(np.int64), np.asarray(coo.data)
        keep = v != 0
        r, c, v = r[keep], c[keep], v[keep]
        if not directed:
            if tri is None:
                dense_like = sp.csr_matrix(a)
                if (dense_like != dense_like.T).nnz:
                    raise MatlabError("Adjacency matrix must be symmetric. Use graph(A, 'upper') or "
                                      "graph(A, 'lower') to use one triangle, or digraph(A).")
                sel = r <= c
            elif tri == "upper":
                sel = r <= c
            else:
                sel = r >= c
            r, c, v = r[sel], c[sel], v[sel]
        src, dst = r, c
        weights = None if logical else np.real(v).astype(float)
        num = n
    if omit_loops:
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if weights is not None:
            weights = weights[keep]
        edge_vars = {k: _take_rows(v, np.nonzero(keep)[0]) for k, v in edge_vars.items()}
    need = int(max(src.max(initial=-1), dst.max(initial=-1))) + 1
    if num is None:
        num = max(need, len(names) if names is not None else 0)
    elif need > num:
        raise MatlabError(f"{kind}: an edge refers to node {need}, but the graph has only {num} nodes.")
    if names is not None:
        names = _check_names(names, num)
    for v in node_vars.values():
        if _col_height(v) != num:
            raise MatlabError(f"{kind}: the node table must have one row per node.")
    return GraphObject(directed, num, src, dst, weights, names, edge_vars, node_vars)


def _adjacency_arg(v: Any, kind: str):
    import scipy.sparse as sp
    if hasattr(v, "to_dense") and hasattr(v, "m"):
        m = v.m
    elif isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        m = sp.csr_matrix(v if v.dtype == bool else v.astype(float))
    else:
        raise MatlabError(f"{kind}: expected an adjacency matrix, an edge list (s, t) or an edge table.")
    if m.shape[0] != m.shape[1]:
        raise MatlabError(f"{kind}: the adjacency matrix must be square.")
    return m


@builtin("graph")
def _graph(interp, args, nargout):
    return _construct(False, args)


@builtin("digraph")
def _digraph(interp, args, nargout):
    return _construct(True, args)


# -- methods are collected here; see the registration at the end of the module ----------------------

_METHODS: dict = {}


def method(*names: str, public: bool = True):
    """Register a graph method. ``public`` ones also become built-ins that check their argument."""
    def deco(fn):
        for n in names:
            _METHODS[n] = fn
            if public:
                _PUBLIC.append(n)
        return fn
    return deco


_PUBLIC: list[str] = []


def _g(args: list, name: str) -> GraphObject:
    if not args or not isinstance(args[0], GraphObject):
        raise MatlabError(f"{name}: the first argument must be a graph or digraph object.")
    return args[0]


def _options(rest: list, allowed: dict, name: str) -> dict:
    """Parse trailing name-value pairs; ``allowed`` maps lower-case names to their canonical form."""
    if len(rest) % 2:
        raise MatlabError(f"{name}: options must be given as name-value pairs.")
    out = {}
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]):
            raise MatlabError(f"{name}: option names must be text.")
        key = text(rest[k]).lower()
        if key not in allowed:
            raise MatlabError(f"{name}: unknown option '{text(rest[k])}'.")
        out[allowed[key]] = rest[k + 1]
    return out


def _choice(v: Any, allowed: tuple, name: str, what: str) -> str:
    s = text(v).lower() if is_text(v) else ""
    if s not in allowed:
        raise MatlabError(f"{name}: {what} must be one of: " + ", ".join(f"'{a}'" for a in allowed) + ".")
    return s


@method("numnodes")
def _numnodes(interp, args, nargout):
    return float(_g(args, "numnodes").n)


@method("numedges")
def _numedges(interp, args, nargout):
    return float(_g(args, "numedges").m)


@method("ismultigraph")
def _ismultigraph(interp, args, nargout):
    g = _g(args, "ismultigraph")
    pairs = set(zip(g.src.tolist(), g.dst.tolist()))
    return bool_scalar(len(pairs) != g.m)


@method("findnode")
def _findnode(interp, args, nargout):
    g = _g(args, "findnode")
    if len(args) != 2:
        raise MatlabError("findnode expects a graph and node IDs or names.")
    v = args[1]
    if _is_names(v):
        idx = g._name_index()
        return _col([idx.get(s, -1) + 1.0 for s in _name_list(v)])
    a = np.real(fnum(v)).reshape(-1, order="F")
    if a.size and (np.any(a != np.round(a)) or np.any(a < 1)):
        raise MatlabError("findnode: node IDs must be positive integers.")
    return _col([x if x <= g.n else 0.0 for x in a])


def _pairs(g: GraphObject, s_v: Any, t_v: Any, name: str):
    s, named = g.node_ids(s_v)
    t, _ = g.node_ids(t_v)
    if s.size == 1 and t.size > 1:
        s = np.full(t.size, s[0])
    if t.size == 1 and s.size > 1:
        t = np.full(s.size, t[0])
    if s.size != t.size:
        raise MatlabError(f"{name}: s and t must have the same number of elements.")
    return s, t, named


def _edge_lookup(g: GraphObject) -> dict:
    table: dict = {}
    for e in range(g.m):
        table.setdefault((int(g.src[e]), int(g.dst[e])), []).append(e)
    return table


@method("findedge")
def _findedge(interp, args, nargout):
    g = _g(args, "findedge")
    if len(args) == 1:
        # all edges as (s, t)
        s, t = g.node_out(g.src, g.names is not None), g.node_out(g.dst, g.names is not None)
        return (s, t) if nargout > 1 else s
    if len(args) == 2:
        idx = np.real(fnum(args[1])).reshape(-1, order="F")
        if idx.size and (np.any(idx != np.round(idx)) or np.any(idx < 1) or np.any(idx > g.m)):
            raise MatlabError("findedge: edge indices must be positive integers no larger than numedges.")
        e = idx.astype(np.int64) - 1
        s, t = g.node_out(g.src[e], g.names is not None), g.node_out(g.dst[e], g.names is not None)
        return (s, t) if nargout > 1 else s
    if len(args) != 3:
        raise MatlabError("findedge expects a graph and either edge indices or node pairs (s, t).")
    s, t, _ = _pairs(g, args[1], args[2], "findedge")
    table = _edge_lookup(g)
    first, every, which = [], [], []
    for k in range(s.size):
        a, b = int(s[k]), int(t[k])
        key = (a, b) if g.directed else (min(a, b), max(a, b))
        hits = table.get(key, [])
        first.append(hits[0] + 1.0 if hits else 0.0)
        if hits:
            every.extend(h + 1.0 for h in hits)
            which.extend([k + 1.0] * len(hits))
        else:
            every.append(0.0)
            which.append(k + 1.0)
    if nargout > 1:
        return _col(every), _col(which)
    return _col(first)


@method("edgecount")
def _edgecount(interp, args, nargout):
    g = _g(args, "edgecount")
    if len(args) != 3:
        raise MatlabError("edgecount expects a graph and node pairs (s, t).")
    s, t, _ = _pairs(g, args[1], args[2], "edgecount")
    table = _edge_lookup(g)
    out = []
    for k in range(s.size):
        a, b = int(s[k]), int(t[k])
        out.append(float(len(table.get((a, b) if g.directed else (min(a, b), max(a, b)), []))))
    return _col(out)


# -- modifying a graph -------------------------------------------------------------------------------

def _grow_nodes(g: GraphObject, count: int, new_names: list[str] | None, extra: dict | None = None) -> GraphObject:
    n2 = g.n + count
    names = g.names
    if new_names is not None:
        if g.names is None and g.n > 0:
            raise MatlabError("Cannot add named nodes to a graph whose nodes have no names.")
        names = _check_names((g.names or []) + new_names, n2)
    elif g.names is not None and count:
        base = [f"Node{k + 1}" for k in range(g.n, n2)]
        names = _check_names(g.names + base, n2)
    node_vars = {}
    extra = extra or {}
    for k, v in g.node_vars.items():
        node_vars[k] = _cat_cols(v, extra[k] if k in extra else _filler(v, count))
    for k, v in extra.items():
        if k not in node_vars:
            if g.n:
                node_vars[k] = _cat_cols(_filler(v, g.n), v)
            else:
                node_vars[k] = v
    return g.clone(n=n2, names=names, node_vars=node_vars)


@method("addnode")
def _addnode(interp, args, nargout):
    g = _g(args, "addnode")
    if len(args) != 2:
        raise MatlabError("addnode expects a graph and the nodes to add (a count, names or a table).")
    v = args[1]
    table_cls = _table_cls()
    if isinstance(v, table_cls):
        nm, rest = _split_table(v, "Name", "Nodes")
        return _grow_nodes(g, v.nrows, _name_list(nm) if nm is not None else None, rest)
    if _is_names(v):
        new = _name_list(v)
        if g.names is not None and any(s in g._name_index() for s in new):
            raise MatlabError("addnode: a node with that name already exists.")
        return _grow_nodes(g, len(new), new)
    count = to_int(v, "the number of nodes")
    if count < 0:
        raise MatlabError("addnode: the number of nodes must be nonnegative.")
    return _grow_nodes(g, count, None)


@method("rmnode")
def _rmnode(interp, args, nargout):
    g = _g(args, "rmnode")
    if len(args) != 2:
        raise MatlabError("rmnode expects a graph and the nodes to remove.")
    ids, _ = g.node_ids(args[1])
    keep = np.ones(g.n, dtype=bool)
    keep[ids] = False
    return _induced(g, np.nonzero(keep)[0])


def _induced(g: GraphObject, nodes: np.ndarray) -> GraphObject:
    """Subgraph on ``nodes`` (0-based, in the order given)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    new_id = -np.ones(g.n, dtype=np.int64)
    new_id[nodes] = np.arange(nodes.size)
    sel = (new_id[g.src] >= 0) & (new_id[g.dst] >= 0) if g.m else np.zeros(0, dtype=bool)
    idx = np.nonzero(sel)[0]
    return GraphObject(
        g.directed, nodes.size, new_id[g.src[idx]], new_id[g.dst[idx]],
        None if g.w is None else g.w[idx],
        None if g.names is None else [g.names[i] for i in nodes],
        {k: _take_rows(v, idx) for k, v in g.edge_vars.items()},
        {k: _take_rows(v, nodes) for k, v in g.node_vars.items()})


@method("subgraph")
def _subgraph(interp, args, nargout):
    g = _g(args, "subgraph")
    if len(args) != 2:
        raise MatlabError("subgraph expects a graph and a list of nodes.")
    v = args[1]
    if isinstance(v, np.ndarray) and v.dtype == bool:
        if v.size != g.n:
            raise MatlabError("subgraph: a logical index must have one element per node.")
        nodes = np.nonzero(v.reshape(-1, order="F"))[0]
    else:
        nodes, _ = g.node_ids(v)
        if np.unique(nodes).size != nodes.size:
            raise MatlabError("subgraph: node IDs must be unique.")
    return _induced(g, nodes)


@method("reordernodes")
def _reordernodes(interp, args, nargout):
    g = _g(args, "reordernodes")
    if len(args) != 2:
        raise MatlabError("reordernodes expects a graph and a permutation of its nodes.")
    order, _ = g.node_ids(args[1])
    if order.size != g.n or np.unique(order).size != g.n:
        raise MatlabError("reordernodes: the order must be a permutation of all node IDs.")
    new_id = np.empty(g.n, dtype=np.int64)
    new_id[order] = np.arange(g.n)
    src, dst = new_id[g.src], new_id[g.dst]
    if not g.directed:
        src, dst = np.minimum(src, dst), np.maximum(src, dst)
    perm = np.lexsort((dst, src)) if g.m else np.zeros(0, dtype=np.int64)
    h = GraphObject(g.directed, g.n, src[perm], dst[perm], None if g.w is None else g.w[perm],
                    None if g.names is None else [g.names[i] for i in order],
                    {k: _take_rows(v, perm) for k, v in g.edge_vars.items()},
                    {k: _take_rows(v, order) for k, v in g.node_vars.items()}, presorted=True)
    return (h, _col(perm + 1.0)) if nargout > 1 else h


@method("addedge")
def _addedge(interp, args, nargout):
    g = _g(args, "addedge")
    table_cls = _table_cls()
    extra_vars: dict = {}
    w_new = None
    if len(args) == 2 and isinstance(args[1], table_cls):
        t = args[1]
        end, rest = _split_table(t, "EndNodes", "Edges")
        if end is None:
            raise MatlabError("addedge: the edge table must have a variable named EndNodes.")
        s_v = Cell(end.data[:, 0:1]) if isinstance(end, Cell) else end[:, 0:1]
        t_v = Cell(end.data[:, 1:2]) if isinstance(end, Cell) else end[:, 1:2]
        if "Weight" in rest:
            w_new = np.real(fnum(rest.pop("Weight"))).astype(float).reshape(-1)
        extra_vars = rest
    elif len(args) in (3, 4):
        s_v, t_v = args[1], args[2]
        if len(args) == 4:
            if isinstance(args[3], table_cls):
                rest = dict(zip(args[3].names, args[3].columns))
                if "Weight" in rest:
                    w_new = np.real(fnum(rest.pop("Weight"))).astype(float).reshape(-1)
                extra_vars = rest
            else:
                w_new = np.real(fnum(args[3])).astype(float).reshape(-1, order="F")
    else:
        raise MatlabError("addedge expects (G, s, t), (G, s, t, weights) or (G, EdgeTable).")
    h = g
    # new nodes are created for unknown names and for indices beyond the current node count
    if _is_names(s_v) or _is_names(t_v):
        if not (_is_names(s_v) and _is_names(t_v)):
            raise MatlabError("addedge: s and t must both be node indices or both be node names.")
        s_names, t_names = _name_list(s_v), _name_list(t_v)
        if g.names is None and g.n > 0:
            raise MatlabError("addedge: the graph has no node names; refer to nodes by their numeric IDs.")
        known = set(g.names or [])
        new = list(dict.fromkeys(x for x in s_names + t_names if x not in known))
        if new:
            h = _grow_nodes(h, len(new), new)
        idx = h._name_index()
        s = np.array([idx[x] for x in s_names], dtype=np.int64)
        t = np.array([idx[x] for x in t_names], dtype=np.int64)
    else:
        sa = np.real(fnum(s_v)).reshape(-1, order="F")
        ta = np.real(fnum(t_v)).reshape(-1, order="F")
        for a in (sa, ta):
            if a.size and (np.any(a != np.round(a)) or np.any(a < 1)):
                raise MatlabError("addedge: node indices must be positive integers.")
        s, t = sa.astype(np.int64) - 1, ta.astype(np.int64) - 1
        need = int(max(s.max(initial=-1), t.max(initial=-1))) + 1
        if need > h.n:
            h = _grow_nodes(h, need - h.n, None)
    if s.size == 1 and t.size > 1:
        s = np.full(t.size, s[0])
    if t.size == 1 and s.size > 1:
        t = np.full(s.size, t[0])
    if s.size != t.size:
        raise MatlabError("addedge: s and t must have the same number of elements.")
    k = s.size
    if w_new is not None:
        if w_new.size == 1 and k != 1:
            w_new = np.full(k, w_new[0])
        if w_new.size != k:
            raise MatlabError("addedge: the number of weights must match the number of new edges.")
        if h.w is None and h.m > 0:
            raise MatlabError("addedge: the graph has no edge weights, so weights cannot be given for new edges.")
    elif h.w is not None:
        raise MatlabError("addedge: the graph has edge weights; give a weight for each new edge.")
    weights = None
    if h.w is not None or w_new is not None:
        weights = np.concatenate([h.w if h.w is not None else np.zeros(0), w_new if w_new is not None else []])
    edge_vars = {}
    for name, col in h.edge_vars.items():
        edge_vars[name] = _cat_cols(col, extra_vars[name] if name in extra_vars else _filler(col, k))
    for name, col in extra_vars.items():
        if name not in edge_vars:
            edge_vars[name] = _cat_cols(_filler(col, h.m), col) if h.m else col
    return GraphObject(h.directed, h.n, np.concatenate([h.src, s]), np.concatenate([h.dst, t]), weights,
                       h.names, edge_vars, h.node_vars)


@method("rmedge")
def _rmedge(interp, args, nargout):
    g = _g(args, "rmedge")
    if len(args) == 2:
        idx = np.real(fnum(args[1])).reshape(-1, order="F")
        if idx.size and (np.any(idx != np.round(idx)) or np.any(idx < 1) or np.any(idx > g.m)):
            raise MatlabError("rmedge: edge indices must be positive integers no larger than numedges.")
        drop = idx.astype(np.int64) - 1
    elif len(args) == 3:
        s, t, _ = _pairs(g, args[1], args[2], "rmedge")
        table = _edge_lookup(g)
        hits: list[int] = []
        for k in range(s.size):
            a, b = int(s[k]), int(t[k])
            hits.extend(table.get((a, b) if g.directed else (min(a, b), max(a, b)), []))
        drop = np.array(hits, dtype=np.int64)
    else:
        raise MatlabError("rmedge expects (G, s, t) or (G, edgeIndices).")
    keep = np.ones(g.m, dtype=bool)
    keep[drop] = False
    idx = np.nonzero(keep)[0]
    return g.clone(src=g.src[idx], dst=g.dst[idx], weights=None if g.w is None else g.w[idx],
                   edge_vars={k: _take_rows(v, idx) for k, v in g.edge_vars.items()})


@method("flipedge")
def _flipedge(interp, args, nargout):
    g = _g(args, "flipedge")
    if not g.directed:
        raise MatlabError("flipedge is only defined for directed graphs (digraph).")
    flip = np.ones(g.m, dtype=bool)
    if len(args) == 2:
        idx = np.real(fnum(args[1])).reshape(-1, order="F").astype(np.int64) - 1
        flip[:] = False
        flip[idx] = True
    elif len(args) == 3:
        s, t, _ = _pairs(g, args[1], args[2], "flipedge")
        table = _edge_lookup(g)
        flip[:] = False
        for k in range(s.size):
            for e in table.get((int(s[k]), int(t[k])), []):
                flip[e] = True
    elif len(args) != 1:
        raise MatlabError("flipedge expects (G), (G, s, t) or (G, edgeIndices).")
    src = np.where(flip, g.dst, g.src)
    dst = np.where(flip, g.src, g.dst)
    return GraphObject(True, g.n, src, dst, g.w, g.names, g.edge_vars, g.node_vars)


# -- neighbours and degrees ----------------------------------------------------------------------------

@method("neighbors")
def _neighbors(interp, args, nargout):
    g = _g(args, "neighbors")
    if g.directed:
        raise MatlabError("neighbors is defined for undirected graphs; use successors or predecessors "
                          "with a digraph.")
    if len(args) != 2:
        raise MatlabError("neighbors expects a graph and one node.")
    u, named = g.one_node(args[1])
    pairs = g.out_lists()[u]
    ids = g.node_out([v for v, _ in pairs], named)
    if nargout > 1:
        return ids, _col([e + 1.0 for _, e in pairs])
    return ids


@method("successors")
def _successors(interp, args, nargout):
    g = _g(args, "successors")
    if not g.directed:
        raise MatlabError("successors is defined for directed graphs; use neighbors with a graph.")
    if len(args) != 2:
        raise MatlabError("successors expects a digraph and one node.")
    u, named = g.one_node(args[1])
    return g.node_out([v for v, _ in g.out_lists()[u]], named)


@method("predecessors")
def _predecessors(interp, args, nargout):
    g = _g(args, "predecessors")
    if not g.directed:
        raise MatlabError("predecessors is defined for directed graphs; use neighbors with a graph.")
    if len(args) != 2:
        raise MatlabError("predecessors expects a digraph and one node.")
    u, named = g.one_node(args[1])
    return g.node_out([v for v, _ in g.in_lists()[u]], named)


def _degrees(g: GraphObject) -> tuple[np.ndarray, np.ndarray]:
    out = np.bincount(g.src, minlength=g.n).astype(float) if g.m else np.zeros(g.n)
    inn = np.bincount(g.dst, minlength=g.n).astype(float) if g.m else np.zeros(g.n)
    return out, inn


def _deg_result(g: GraphObject, d: np.ndarray, args: list):
    if len(args) > 2:
        raise MatlabError("Too many input arguments.")
    if len(args) == 2:
        ids, _ = g.node_ids(args[1])
        shape = args[1].shape if isinstance(args[1], np.ndarray) and not is_char(args[1]) else (ids.size, 1)
        return d[ids].reshape(shape, order="F")
    return d.reshape(-1, 1)


@method("degree")
def _degree(interp, args, nargout):
    g = _g(args, "degree")
    if g.directed:
        raise MatlabError("degree is defined for undirected graphs; use indegree and outdegree with a digraph.")
    out, inn = _degrees(g)
    return _deg_result(g, out + inn, args)  # a self-loop counts twice


@method("indegree")
def _indegree(interp, args, nargout):
    g = _g(args, "indegree")
    if not g.directed:
        raise MatlabError("indegree is defined for directed graphs; use degree with a graph.")
    return _deg_result(g, _degrees(g)[1], args)


@method("outdegree")
def _outdegree(interp, args, nargout):
    g = _g(args, "outdegree")
    if not g.directed:
        raise MatlabError("outdegree is defined for directed graphs; use degree with a graph.")
    return _deg_result(g, _degrees(g)[0], args)


def _incident(g: GraphObject, args: list, name: str, incoming: bool, nargout: int):
    if len(args) != 2:
        raise MatlabError(f"{name} expects a graph and one node.")
    u, named = g.one_node(args[1])
    if g.directed:
        pairs = sorted(((e, v) for v, e in (g.in_lists() if incoming else g.out_lists())[u]))
    else:
        pairs = sorted(((e, v) for v, e in g.out_lists()[u]))
    eid = _col([e + 1.0 for e, _ in pairs])
    if nargout > 1:
        return eid, g.node_out([v for _, v in pairs], named)
    return eid


@method("outedges")
def _outedges(interp, args, nargout):
    return _incident(_g(args, "outedges"), args, "outedges", False, nargout)


@method("inedges")
def _inedges(interp, args, nargout):
    g = _g(args, "inedges")
    if not g.directed:
        raise MatlabError("inedges is defined for directed graphs; use outedges with a graph.")
    return _incident(g, args, "inedges", True, nargout)


# -- matrices ---------------------------------------------------------------------------------------------

def _sparse(m):
    from .sparse import SparseMatrix
    return SparseMatrix(m)


@method("adjacency")
def _adjacency(interp, args, nargout):
    import scipy.sparse as sp
    g = _g(args, "adjacency")
    if len(args) > 2:
        raise MatlabError("Too many input arguments for adjacency.")
    vals = np.ones(g.m)
    if len(args) == 2:
        if is_text(args[1]):
            if text(args[1]).lower() != "weighted":
                raise MatlabError("adjacency: the second argument must be 'weighted' or a vector of edge weights.")
            if g.w is None:
                raise MatlabError("adjacency: the graph has no edge weights.")
            vals = g.w
        else:
            vals = np.real(fnum(args[1])).astype(float).reshape(-1, order="F")
            if vals.size != g.m:
                raise MatlabError("adjacency: the weight vector must have one element per edge.")
    rows, cols, data = g.src, g.dst, vals
    if not g.directed:
        off = g.src != g.dst
        rows = np.concatenate([g.src, g.dst[off]])
        cols = np.concatenate([g.dst, g.src[off]])
        data = np.concatenate([vals, vals[off]])
    return _sparse(sp.csc_matrix((data, (rows, cols)), shape=(g.n, g.n)))


@method("incidence")
def _incidence(interp, args, nargout):
    import scipy.sparse as sp
    g = _g(args, "incidence")
    if len(args) != 1:
        raise MatlabError("Too many input arguments for incidence.")
    e = np.arange(g.m)
    rows = np.concatenate([g.src, g.dst])
    cols = np.concatenate([e, e])
    data = np.concatenate([-np.ones(g.m), np.ones(g.m)])
    return _sparse(sp.csc_matrix((data, (rows, cols)), shape=(g.n, g.m)))


@method("laplacian", public=False)
def _laplacian(interp, args, nargout):
    import scipy.sparse as sp
    g = _g(args, "laplacian")
    if g.directed:
        raise MatlabError("laplacian is only defined for undirected graphs.")
    if len(args) != 1:
        raise MatlabError("Too many input arguments for laplacian.")
    if np.any(g.src == g.dst) or len(set(zip(g.src.tolist(), g.dst.tolist()))) != g.m:
        raise MatlabError("laplacian: the graph must not have self-loops or multiple edges.")
    rows = np.concatenate([g.src, g.dst, np.arange(g.n)])
    cols = np.concatenate([g.dst, g.src, np.arange(g.n)])
    deg = np.bincount(np.concatenate([g.src, g.dst]), minlength=g.n).astype(float) if g.m else np.zeros(g.n)
    data = np.concatenate([-np.ones(2 * g.m), deg])
    return _sparse(sp.csc_matrix((data, (rows, cols)), shape=(g.n, g.n)))


# -- shortest paths ------------------------------------------------------------------------------------------

_SP_METHODS = ("auto", "unweighted", "positive", "mixed", "acyclic")


def _sp_setup(g: GraphObject, method: str, name: str) -> tuple[np.ndarray, str]:
    """Edge costs and the algorithm ('bfs', 'dijkstra', 'bellman' or 'dag') for a Method option."""
    if method == "unweighted" or (method == "auto" and g.w is None):
        return np.ones(g.m), "bfs"
    w = g.weights()
    if np.any(np.isnan(w)):
        raise MatlabError(f"{name}: edge weights must not be NaN.")
    if method == "auto":
        method = "positive" if np.all(w >= 0) else "mixed"
    if method == "positive":
        if np.any(w < 0):
            raise MatlabError(f"{name}: the 'positive' method needs nonnegative edge weights.")
        return w, "dijkstra"
    if method == "mixed":
        if not g.directed and np.any(w < 0):
            raise MatlabError(f"{name}: an undirected graph with negative edge weights has no shortest paths "
                              "(every negative edge is a negative cycle).")
        return w, "bellman"
    if not g.directed or not _is_dag(g):
        raise MatlabError(f"{name}: the 'acyclic' method needs a directed acyclic graph.")
    return w, "dag"


def _sssp(g: GraphObject, source: int, cost: np.ndarray, alg: str, reverse: bool = False):
    """Single-source shortest paths. Returns (dist, predecessor node, predecessor edge)."""
    n = g.n
    lists = g.in_lists() if reverse else g.out_lists()
    dist = np.full(n, np.inf)
    pred = np.full(n, -1, dtype=np.int64)
    pedge = np.full(n, -1, dtype=np.int64)
    dist[source] = 0.0
    if alg == "bfs":
        queue = [source]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v, e in lists[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1.0
                    pred[v], pedge[v] = u, e
                    queue.append(v)
    elif alg == "dijkstra":
        done = np.zeros(n, dtype=bool)
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            for v, e in lists[u]:
                nd = d + cost[e]
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v], pedge[v] = u, e
                    heapq.heappush(heap, (nd, v))
    elif alg == "dag":
        order = _topo_order(g, stable=False)
        if reverse:
            order = order[::-1]
        for u in order:
            if dist[u] == np.inf:
                continue
            for v, e in lists[u]:
                nd = dist[u] + cost[e]
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v], pedge[v] = u, e
    else:  # Bellman-Ford
        for _ in range(n):
            changed = False
            for u in range(n):
                if dist[u] == np.inf:
                    continue
                for v, e in lists[u]:
                    nd = dist[u] + cost[e]
                    if nd < dist[v]:
                        dist[v] = nd
                        pred[v], pedge[v] = u, e
                        changed = True
            if not changed:
                break
        else:
            raise MatlabError("The graph contains a negative cycle, so shortest paths are not defined.")
    return dist, pred, pedge


def _walk(pred: np.ndarray, pedge: np.ndarray, target: int) -> tuple[list[int], list[int]]:
    nodes, edges = [target], []
    v = target
    while pred[v] >= 0:
        edges.append(int(pedge[v]))
        v = int(pred[v])
        nodes.append(v)
    return nodes[::-1], edges[::-1]


def _method_opt(rest: list, name: str, extra: dict | None = None) -> dict:
    allowed = {"method": "Method"}
    allowed.update(extra or {})
    opts = _options(rest, allowed, name)
    opts["Method"] = _choice(opts["Method"], _SP_METHODS, name, "Method") if "Method" in opts else "auto"
    return opts


@method("shortestpath")
def _shortestpath(interp, args, nargout):
    g = _g(args, "shortestpath")
    if len(args) < 3:
        raise MatlabError("shortestpath expects a graph, a source node and a target node.")
    s, named = g.one_node(args[1], "source node")
    t, _ = g.one_node(args[2], "target node")
    opts = _method_opt(args[3:], "shortestpath")
    cost, alg = _sp_setup(g, opts["Method"], "shortestpath")
    dist, pred, pedge = _sssp(g, s, cost, alg)
    if not np.isfinite(dist[t]):
        path = _names_cell([], column=False) if named else np.zeros((1, 0))
        return (path, np.inf, np.zeros((1, 0)))[:max(nargout, 1)] if nargout > 1 else path
    nodes, edges = _walk(pred, pedge, t)
    path = g.node_out(nodes, named, column=False)
    outs = (path, float(dist[t]), _row([e + 1.0 for e in edges]))
    return outs[:max(nargout, 1)] if nargout > 1 else path


def _node_list_or_all(g: GraphObject, v: Any) -> tuple[np.ndarray, bool, bool]:
    if is_text(v) and not isinstance(v, Cell) and text(v).lower() == "all" and \
            (g.names is None or "all" not in g._name_index()):
        return np.arange(g.n, dtype=np.int64), False, True
    ids, named = g.node_ids(v)
    return ids, named, False


@method("distances")
def _distances(interp, args, nargout):
    g = _g(args, "distances")
    rest = list(args[1:])
    pos = []
    while rest and not (is_text(rest[0]) and not isinstance(rest[0], Cell)
                        and text(rest[0]).lower() == "method" and len(rest) % 2 == 0):
        pos.append(rest.pop(0))
        if len(pos) == 2:
            break
    opts = _method_opt(rest, "distances")
    src = _node_list_or_all(g, pos[0])[0] if pos else np.arange(g.n, dtype=np.int64)
    dst = _node_list_or_all(g, pos[1])[0] if len(pos) > 1 else np.arange(g.n, dtype=np.int64)
    cost, alg = _sp_setup(g, opts["Method"], "distances")
    out = np.zeros((src.size, dst.size))
    for k, s in enumerate(src):
        out[k, :] = _sssp(g, int(s), cost, alg)[0][dst]
    return out


@method("shortestpathtree")
def _shortestpathtree(interp, args, nargout):
    g = _g(args, "shortestpathtree")
    if len(args) < 2:
        raise MatlabError("shortestpathtree expects a graph and a source node.")
    rest = list(args[2:])
    target_v = None
    if len(rest) % 2 == 1:
        target_v = rest.pop(0)
    opts = _method_opt(rest, "shortestpathtree", {"outputform": "OutputForm"})
    form = _choice(opts["OutputForm"], ("tree", "cell", "vector"), "shortestpathtree", "OutputForm") \
        if "OutputForm" in opts else "tree"
    s_ids, s_named, s_all = _node_list_or_all(g, args[1])
    if target_v is None:
        t_ids, t_named, t_all = np.arange(g.n, dtype=np.int64), False, True
    else:
        t_ids, t_named, t_all = _node_list_or_all(g, target_v)
    named = s_named or t_named
    reverse = False
    if s_ids.size != 1:
        if t_ids.size != 1:
            raise MatlabError("shortestpathtree: either the source or the target must be a single node.")
        root, others, reverse = int(t_ids[0]), s_ids, True  # all paths lead to one target
    else:
        root, others = int(s_ids[0]), t_ids
    cost, alg = _sp_setup(g, opts["Method"], "shortestpathtree")
    dist, pred, pedge = _sssp(g, root, cost, alg, reverse=reverse)
    d_out = _row(dist[others])
    if form == "cell":
        paths, epaths = [], []
        for v in others:
            if np.isfinite(dist[v]):
                nodes, edges = _walk(pred, pedge, int(v))
                if reverse:
                    nodes, edges = nodes[::-1], edges[::-1]
            else:
                nodes, edges = [], []
            paths.append(g.node_out(nodes, named, column=False))
            epaths.append(_row([e + 1.0 for e in edges]))
        outs = (Cell.from_list(paths, column=True), d_out, Cell.from_list(epaths, column=True))
        return outs[:max(nargout, 1)] if nargout > 1 else outs[0]
    # nodes on the paths to the requested targets
    keep = np.zeros(g.n, dtype=bool)
    for v in others:
        v = int(v)
        while np.isfinite(dist[v]) and not keep[v]:
            keep[v] = True
            if pred[v] < 0:
                break
            v = int(pred[v])
    in_tree = np.zeros(g.m, dtype=bool)
    for v in range(g.n):
        if keep[v] and pedge[v] >= 0:
            in_tree[pedge[v]] = True
    if form == "vector":
        pv = np.where(pred >= 0, pred + 1.0, np.where(np.arange(g.n) == root, 0.0, np.nan))
        ev = np.where(pedge >= 0, pedge + 1.0, np.where(np.arange(g.n) == root, 0.0, np.nan))
        pv = np.where(keep | (np.arange(g.n) == root), pv, np.nan)
        ev = np.where(keep | (np.arange(g.n) == root), ev, np.nan)
        outs = (_row(pv), d_out, _row(ev))
        return outs[:max(nargout, 1)] if nargout > 1 else outs[0]
    idx = np.nonzero(in_tree)[0]
    # tree edges point away from the source (or toward the single target)
    child = np.array([v for v in range(g.n) if keep[v] and pedge[v] >= 0], dtype=np.int64)
    by_edge = {int(pedge[v]): v for v in child}
    src, dst = [], []
    for e in idx:
        v = by_edge[int(e)]
        u = int(pred[v])
        src.append(v if reverse else u)
        dst.append(u if reverse else v)
    tree = GraphObject(True, g.n, src, dst, None if g.w is None else g.w[idx], g.names,
                       {k: _take_rows(c, idx) for k, c in g.edge_vars.items()}, g.node_vars)
    outs = (tree, d_out, in_tree.reshape(-1, 1))
    return outs[:max(nargout, 1)] if nargout > 1 else tree


@method("nearest")
def _nearest(interp, args, nargout):
    g = _g(args, "nearest")
    if len(args) < 3:
        raise MatlabError("nearest expects a graph, a source node and a distance.")
    s, named = g.one_node(args[1])
    radius = float(np.real(fnum(args[2])).flat[0])
    opts = _method_opt(args[3:], "nearest", {"direction": "Direction"})
    direction = _choice(opts["Direction"], ("outgoing", "incoming"), "nearest", "Direction") \
        if "Direction" in opts else "outgoing"
    cost, alg = _sp_setup(g, opts["Method"], "nearest")
    dist = _sssp(g, s, cost, alg, reverse=(direction == "incoming" and g.directed))[0]
    ids = [v for v in range(g.n) if v != s and dist[v] <= radius]
    ids.sort(key=lambda v: (dist[v], v))
    out = g.node_out(ids, named)
    return (out, _col(dist[ids])) if nargout > 1 else out


# -- connectivity and traversal ---------------------------------------------------------------------------------

def _weak_components(g: GraphObject) -> np.ndarray:
    parent = list(range(g.n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in range(g.m):
        a, b = find(int(g.src[e])), find(int(g.dst[e]))
        if a != b:
            parent[max(a, b)] = min(a, b)
    bins = np.zeros(g.n, dtype=np.int64)
    label: dict[int, int] = {}
    for v in range(g.n):
        bins[v] = label.setdefault(find(v), len(label))
    return bins


def _strong_components(g: GraphObject) -> np.ndarray:
    """Tarjan's algorithm (iterative). Components are numbered in topological order."""
    n = g.n
    adj = g.out_lists()
    index = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    on_stack = np.zeros(n, dtype=bool)
    comp = np.full(n, -1, dtype=np.int64)
    stack: list[int] = []
    counter = 0
    ncomp = 0
    for root in range(n):
        if index[root] >= 0:
            continue
        work = [(root, 0)]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            u, k = work[-1]
            if k < len(adj[u]):
                work[-1] = (u, k + 1)
                v = adj[u][k][0]
                if index[v] < 0:
                    index[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack[v] = True
                    work.append((v, 0))
                elif on_stack[v]:
                    low[u] = min(low[u], index[v])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[u])
                if low[u] == index[u]:
                    while True:
                        v = stack.pop()
                        on_stack[v] = False
                        comp[v] = ncomp
                        if v == u:
                            break
                    ncomp += 1
    # Tarjan finds sink components first: reverse the numbering for a topological order
    return (ncomp - 1) - comp


@method("conncomp")
def _conncomp(interp, args, nargout):
    g = _g(args, "conncomp")
    opts = _options(args[1:], {"type": "Type", "outputform": "OutputForm"}, "conncomp")
    kind = _choice(opts["Type"], ("strong", "weak"), "conncomp", "Type") if "Type" in opts else \
        ("strong" if g.directed else "weak")
    if "Type" in opts and not g.directed:
        raise MatlabError("conncomp: the Type option applies only to directed graphs.")
    form = _choice(opts["OutputForm"], ("vector", "cell"), "conncomp", "OutputForm") \
        if "OutputForm" in opts else "vector"
    bins = _strong_components(g) if (g.directed and kind == "strong") else _weak_components(g)
    nb = int(bins.max()) + 1 if g.n else 0
    sizes = _row(np.bincount(bins, minlength=nb)) if g.n else np.zeros((1, 0))
    if form == "cell":
        groups = [g.node_out(np.nonzero(bins == b)[0], g.names is not None, column=False) for b in range(nb)]
        out = Cell.from_list(groups)
    else:
        out = _row(bins + 1.0)
    return (out, sizes) if nargout > 1 else out


_EVENTS = ("discovernode", "finishnode", "startnode", "edgetonew", "edgetodiscovered", "edgetofinished")


def _search(g: GraphObject, start: int, restart: bool, depth_first: bool) -> list[tuple]:
    """Event list of a breadth-first or depth-first search: (event, node, (u, v), edge index)."""
    n = g.n
    adj = g.out_lists()
    state = np.zeros(n, dtype=np.int8)  # 0 new, 1 discovered, 2 finished
    used = np.zeros(g.m, dtype=bool)
    events: list[tuple] = []

    def classify(u, v, e):
        if state[v] == 0:
            events.append(("edgetonew", -1, (u, v), e))
            return True
        events.append(("edgetodiscovered" if state[v] == 1 else "edgetofinished", -1, (u, v), e))
        return False

    def run(s):
        events.append(("startnode", s, None, -1))
        state[s] = 1
        events.append(("discovernode", s, None, -1))
        if depth_first:
            stack = [(s, 0)]
            while stack:
                u, k = stack[-1]
                if k < len(adj[u]):
                    stack[-1] = (u, k + 1)
                    v, e = adj[u][k]
                    if not g.directed:
                        if used[e]:
                            continue
                        used[e] = True
                    if classify(u, v, e):
                        state[v] = 1
                        events.append(("discovernode", v, None, -1))
                        stack.append((v, 0))
                else:
                    stack.pop()
                    state[u] = 2
                    events.append(("finishnode", u, None, -1))
        else:
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v, e in adj[u]:
                    if not g.directed:
                        if used[e]:
                            continue
                        used[e] = True
                    if classify(u, v, e):
                        state[v] = 1
                        events.append(("discovernode", v, None, -1))
                        queue.append(v)
                state[u] = 2
                events.append(("finishnode", u, None, -1))

    run(start)
    if restart:
        for s in range(n):
            if state[s] == 0:
                run(s)
    return events


def _search_fn(name: str, depth_first: bool):
    def f(interp, args, nargout):
        g = _g(args, name)
        if len(args) < 2:
            raise MatlabError(f"{name} expects a graph and a starting node.")
        s, named = g.one_node(args[1], "starting node")
        rest = list(args[2:])
        wanted: list[str] = ["discovernode"]
        single = True
        if len(rest) % 2 == 1:
            ev = rest.pop(0)
            if isinstance(ev, (Cell, StringArray)):
                wanted = [x.lower() for x in _name_list(ev)]
                single = False
            else:
                key = text(ev).lower() if is_text(ev) else ""
                if key == "allevents":
                    wanted, single = list(_EVENTS), False
                else:
                    wanted = [key]
            for w in wanted:
                if w not in _EVENTS:
                    raise MatlabError(f"{name}: unknown event '{w}'.")
        opts = _options(rest, {"restart": "Restart"}, name)
        restart = bool(np.all(np.asarray(fnum(opts["Restart"])) != 0)) if "Restart" in opts else False
        events = [ev for ev in _search(g, s, restart, depth_first) if ev[0] in wanted]
        if single:
            if wanted[0] in ("discovernode", "finishnode", "startnode"):
                return g.node_out([ev[1] for ev in events], named)
            pairs = np.array([[ev[2][0] + 1.0, ev[2][1] + 1.0] for ev in events]).reshape(-1, 2)
            eidx = _col([ev[3] + 1.0 for ev in events])
            if named and g.names is not None:
                data = np.empty((len(events), 2), dtype=object)
                for k, ev in enumerate(events):
                    data[k, 0], data[k, 1] = char(g.names[ev[2][0]]), char(g.names[ev[2][1]])
                return (Cell(data), eidx) if nargout > 1 else Cell(data)
            return (pairs, eidx) if nargout > 1 else pairs
        labels = Cell.from_list([char(ev[0]) for ev in events], column=True)
        try:
            event_col = interp.call_named("categorical", [labels, Cell.from_list([char(e) for e in _EVENTS])],
                                          1, interp.frame)[0]
        except MatlabError:
            event_col = labels
        node = _col([ev[1] + 1.0 if ev[1] >= 0 else np.nan for ev in events])
        edge = np.array([[ev[2][0] + 1.0, ev[2][1] + 1.0] if ev[2] else [np.nan, np.nan]
                         for ev in events]).reshape(-1, 2)
        eidx = _col([ev[3] + 1.0 if ev[3] >= 0 else np.nan for ev in events])
        return _table_cls()(["Event", "Node", "Edge", "EdgeIndex"], [event_col, node, edge, eidx])
    return f


method("bfsearch")(_search_fn("bfsearch", False))
method("dfsearch")(_search_fn("dfsearch", True))


def _topo_order(g: GraphObject, stable: bool) -> list[int]:
    """A topological order of a directed graph, or an error when it has a cycle."""
    n = g.n
    adj = g.out_lists()
    if stable:
        # smallest available node first (the lexicographically smallest order)
        indeg = np.bincount(g.dst, minlength=n) if g.m else np.zeros(n, dtype=np.int64)
        heap = [v for v in range(n) if indeg[v] == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            u = heapq.heappop(heap)
            order.append(u)
            for v, _ in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(heap, v)
        if len(order) != n:
            raise MatlabError("The graph has cycles, so it has no topological order.")
        return order
    state = np.zeros(n, dtype=np.int8)
    post: list[int] = []
    for root in range(n):
        if state[root]:
            continue
        stack = [(root, 0)]
        state[root] = 1
        while stack:
            u, k = stack[-1]
            if k < len(adj[u]):
                stack[-1] = (u, k + 1)
                v = adj[u][k][0]
                if state[v] == 0:
                    state[v] = 1
                    stack.append((v, 0))
                elif state[v] == 1:
                    raise MatlabError("The graph has cycles, so it has no topological order.")
            else:
                stack.pop()
                state[u] = 2
                post.append(u)
    return post[::-1]


def _is_dag(g: GraphObject) -> bool:
    try:
        _topo_order(g, stable=True)
        return True
    except MatlabError:
        return False


@method("toposort")
def _toposort(interp, args, nargout):
    g = _g(args, "toposort")
    if not g.directed:
        raise MatlabError("toposort is only defined for directed graphs (digraph).")
    opts = _options(args[1:], {"order": "Order"}, "toposort")
    how = _choice(opts["Order"], ("fast", "stable"), "toposort", "Order") if "Order" in opts else "fast"
    order = np.array(_topo_order(g, stable=(how == "stable")), dtype=np.int64)
    out = _row(order + 1.0)
    if nargout > 1:
        h = _reordernodes(interp, [g, _row(order + 1.0)], 1)
        return out, h
    return out


@method("isdag")
def _isdag(interp, args, nargout):
    g = _g(args, "isdag")
    if not g.directed:
        raise MatlabError("isdag is only defined for directed graphs (digraph).")
    return bool_scalar(_is_dag(g))


@method("hascycles")
def _hascycles(interp, args, nargout):
    g = _g(args, "hascycles")
    if g.directed:
        return bool_scalar(not _is_dag(g))
    # an undirected graph is acyclic exactly when it is a forest
    ncomp = int(_weak_components(g).max()) + 1 if g.n else 0
    return bool_scalar(g.m != g.n - ncomp)


@method("minspantree")
def _minspantree(interp, args, nargout):
    g = _g(args, "minspantree")
    if g.directed:
        raise MatlabError("minspantree is only defined for undirected graphs.")
    opts = _options(args[1:], {"method": "Method", "root": "Root", "type": "Type"}, "minspantree")
    how = _choice(opts["Method"], ("dense", "sparse"), "minspantree", "Method") if "Method" in opts else "dense"
    kind = _choice(opts["Type"], ("tree", "forest"), "minspantree", "Type") if "Type" in opts else "tree"
    root = g.one_node(opts["Root"], "root node")[0] if "Root" in opts else 0
    if g.n == 0:
        return (g, np.zeros((1, 0))) if nargout > 1 else g
    w = g.weights()
    if np.any(np.isnan(w)):
        raise MatlabError("minspantree: edge weights must not be NaN.")
    pred = np.full(g.n, np.nan)
    chosen: list[int] = []
    adj = g.out_lists()
    in_tree = np.zeros(g.n, dtype=bool)

    def prim(start):
        in_tree[start] = True
        pred[start] = 0.0
        heap = [(w[e], e, start, v) for v, e in adj[start]]
        heapq.heapify(heap)
        while heap:
            _, e, u, v = heapq.heappop(heap)
            if in_tree[v]:
                continue
            in_tree[v] = True
            pred[v] = u + 1.0
            chosen.append(e)
            for x, e2 in adj[v]:
                if not in_tree[x]:
                    heapq.heappush(heap, (w[e2], e2, v, x))

    if how == "sparse":
        # Kruskal: scan the edges by weight and join components
        parent = list(range(g.n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for e in np.argsort(w, kind="stable"):
            a, b = find(int(g.src[e])), find(int(g.dst[e]))
            if a != b:
                parent[a] = b
                chosen.append(int(e))
        comp = _weak_components(g)
        keep_comp = None if kind == "forest" else comp[root]
        if keep_comp is not None:
            chosen = [e for e in chosen if comp[g.src[e]] == keep_comp]
        # predecessors from a search of the chosen edges, rooted at the root (and each other tree)
        tree = g.clone(src=g.src[sorted(chosen)], dst=g.dst[sorted(chosen)], weights=None, edge_vars={})
        starts = [root] + ([v for v in range(g.n)] if kind == "forest" else [])
        seen = np.zeros(g.n, dtype=bool)
        for s in starts:
            if seen[s]:
                continue
            seen[s] = True
            pred[s] = 0.0
            queue = [s]
            while queue:
                u = queue.pop()
                for v, _ in tree.out_lists()[u]:
                    if not seen[v]:
                        seen[v] = True
                        pred[v] = u + 1.0
                        queue.append(v)
    else:
        prim(root)
        if kind == "forest":
            for s in range(g.n):
                if not in_tree[s]:
                    prim(s)
    idx = np.array(sorted(chosen), dtype=np.int64)
    t = g.clone(src=g.src[idx], dst=g.dst[idx], weights=None if g.w is None else g.w[idx],
                edge_vars={k: _take_rows(v, idx) for k, v in g.edge_vars.items()})
    return (t, _row(pred)) if nargout > 1 else t


# -- flows ------------------------------------------------------------------------------------------------

@method("maxflow")
def _maxflow(interp, args, nargout):
    g = _g(args, "maxflow")
    if len(args) < 3 or len(args) > 4:
        raise MatlabError("maxflow expects a graph, a source node, a target node and an optional algorithm.")
    s, _ = g.one_node(args[1], "source node")
    t, _ = g.one_node(args[2], "target node")
    if len(args) > 3:
        alg = _choice(args[3], ("searchtrees", "augmentpath", "pushrelabel"), "maxflow", "the algorithm")
        if alg == "pushrelabel" and not g.directed:
            raise MatlabError("maxflow: the 'pushrelabel' algorithm is only available for directed graphs.")
    cap = g.weights()
    if np.any(cap < 0) or np.any(np.isnan(cap)):
        raise MatlabError("maxflow: edge capacities (weights) must be nonnegative.")
    n, m = g.n, g.m
    if s == t:
        empty = GraphObject(True, n, [], [], np.zeros(0) if g.w is not None else None, g.names)
        outs = (0.0, empty, np.zeros((0, 1)), np.zeros((0, 1)))
        return outs[:max(nargout, 1)] if nargout > 1 else 0.0
    # residual network: arc 2e is edge e forward, arc 2e+1 its reverse
    head = np.column_stack([g.dst, g.src]).reshape(-1) if m else np.zeros(0, dtype=np.int64)
    res = np.zeros(2 * m)
    res[0::2] = cap
    res[1::2] = cap if not g.directed else 0.0
    arcs: list = [[] for _ in range(n)]
    for e in range(m):
        arcs[int(g.src[e])].append(2 * e)
        arcs[int(g.dst[e])].append(2 * e + 1)
    total = 0.0
    while True:  # shortest augmenting paths (Edmonds-Karp)
        prev = np.full(n, -1, dtype=np.int64)
        prev[s] = -2
        queue = [s]
        head_i = 0
        while head_i < len(queue) and prev[t] == -1:
            u = queue[head_i]
            head_i += 1
            for a in arcs[u]:
                v = int(head[a])
                if prev[v] == -1 and res[a] > 1e-14 * max(1.0, cap.max(initial=1.0)):
                    prev[v] = a
                    queue.append(v)
        if prev[t] == -1:
            break
        push = np.inf
        v = t
        while v != s:
            a = int(prev[v])
            push = min(push, res[a])
            v = int(head[a ^ 1])
        v = t
        while v != s:
            a = int(prev[v])
            res[a] -= push
            res[a ^ 1] += push
            v = int(head[a ^ 1])
        total += push
    if nargout <= 1:
        return float(total)
    # net flow on every edge, in the direction it runs
    if g.directed:
        flow = cap - res[0::2]
    else:
        flow = (res[1::2] - res[0::2]) / 2.0
    src = np.where(flow >= 0, g.src, g.dst)
    dst = np.where(flow >= 0, g.dst, g.src)
    used = np.abs(flow) > 1e-12 * max(1.0, cap.max(initial=1.0))
    gf = GraphObject(True, n, src[used], dst[used], np.abs(flow)[used], g.names, None, g.node_vars)
    reach = np.zeros(n, dtype=bool)
    reach[queue] = True
    cs = g.node_out(np.nonzero(reach)[0], g.names is not None)
    ct = g.node_out(np.nonzero(~reach)[0], g.names is not None)
    return (float(total), gf, cs, ct)[:nargout]


# -- centrality ---------------------------------------------------------------------------------------------

def _brandes(g: GraphObject, cost: np.ndarray | None) -> np.ndarray:
    """Betweenness: for every node, the number of shortest paths between other nodes through it."""
    n = g.n
    adj = g.out_lists()
    score = np.zeros(n)
    for s in range(n):
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, np.inf)
        dist[s] = 0.0
        preds: list = [[] for _ in range(n)]
        order: list[int] = []
        if cost is None:
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                order.append(u)
                for v, _ in adj[u]:
                    if dist[v] == np.inf:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    if dist[v] == dist[u] + 1:
                        sigma[v] += sigma[u]
                        preds[v].append(u)
        else:
            done = np.zeros(n, dtype=bool)
            heap = [(0.0, s)]
            while heap:
                d, u = heapq.heappop(heap)
                if done[u]:
                    continue
                done[u] = True
                order.append(u)
                for v, e in adj[u]:
                    nd = d + cost[e]
                    if nd < dist[v] * (1 - 1e-12) - 1e-300:
                        dist[v] = nd
                        sigma[v] = sigma[u]
                        preds[v] = [u]
                        heapq.heappush(heap, (nd, v))
                    elif abs(nd - dist[v]) <= 1e-12 * max(abs(nd), 1e-300) and not done[v]:
                        sigma[v] += sigma[u]
                        preds[v].append(u)
        delta = np.zeros(n)
        for v in reversed(order):
            for u in preds[v]:
                delta[u] += sigma[u] / sigma[v] * (1.0 + delta[v])
            if v != s:
                score[v] += delta[v]
    return score if g.directed else score / 2.0


def _closeness(g: GraphObject, cost: np.ndarray, alg: str, reverse: bool) -> np.ndarray:
    n = g.n
    out = np.zeros(n)
    for s in range(n):
        d = _sssp(g, s, cost, alg, reverse=reverse)[0]
        reach = np.isfinite(d)
        reach[s] = False
        a = int(reach.sum())
        if a and n > 1:
            total = float(d[reach].sum())
            out[s] = (a / (n - 1.0)) ** 2 / total if total > 0 else np.inf
    return out


def _pagerank(g: GraphObject, follow: float, tol: float, maxit: int, weights: np.ndarray | None) -> np.ndarray:
    n = g.n
    if n == 0:
        return np.zeros(0)
    w = np.ones(g.m) if weights is None else weights
    src, dst = g.src, g.dst
    if not g.directed:
        off = g.src != g.dst
        src = np.concatenate([g.src, g.dst[off]])
        dst = np.concatenate([g.dst, g.src[off]])
        w = np.concatenate([w, w[off]])
    out_w = np.bincount(src, weights=w, minlength=n) if src.size else np.zeros(n)
    dangling = out_w == 0
    r = np.full(n, 1.0 / n)
    for _ in range(maxit):
        share = np.where(dangling, 0.0, r / np.where(dangling, 1.0, out_w))
        new = np.bincount(dst, weights=share[src] * w, minlength=n) if src.size else np.zeros(n)
        new = follow * (new + r[dangling].sum() / n) + (1 - follow) / n
        done = np.max(np.abs(new - r)) < tol
        r = new
        if done:
            break
    return r / r.sum()


def _power_scores(mat: np.ndarray, tol: float, maxit: int) -> np.ndarray:
    n = mat.shape[0]
    if n == 0:
        return np.zeros(0)
    x = np.full(n, 1.0 / n)
    for _ in range(maxit):
        y = mat @ x
        total = y.sum()
        if total == 0:
            return np.zeros(n)
        y = y / total
        done = np.max(np.abs(y - x)) < tol
        x = y
        if done:
            break
    return x


@method("centrality")
def _centrality(interp, args, nargout):
    g = _g(args, "centrality")
    if len(args) < 2:
        raise MatlabError("centrality expects a graph and a centrality type.")
    kinds = ("indegree", "outdegree", "incloseness", "outcloseness", "betweenness", "pagerank", "hubs",
             "authorities") if g.directed else ("degree", "closeness", "betweenness", "pagerank", "eigenvector")
    kind = _choice(args[1], kinds, "centrality", "the centrality type for a " + g.class_name())
    opts = _options(args[2:], {"importance": "Importance", "cost": "Cost",
                                "followprobability": "FollowProbability", "tolerance": "Tolerance",
                                "maxiterations": "MaxIterations"}, "centrality")

    def edge_vec(key):
        if key not in opts:
            return None
        v = np.real(fnum(opts[key])).astype(float).reshape(-1, order="F")
        if v.size != g.m:
            raise MatlabError(f"centrality: {key} must have one element per edge.")
        return v

    importance, cost = edge_vec("Importance"), edge_vec("Cost")
    uses_cost = kind in ("closeness", "incloseness", "outcloseness", "betweenness")
    if cost is not None and not uses_cost:
        raise MatlabError(f"centrality: the Cost option does not apply to '{kind}'.")
    if importance is not None and uses_cost:
        raise MatlabError(f"centrality: the Importance option does not apply to '{kind}'.")
    if cost is not None and np.any(cost <= 0):
        raise MatlabError("centrality: edge costs must be positive.")
    if importance is not None and np.any(importance < 0):
        raise MatlabError("centrality: edge importances must be nonnegative.")
    iterative = kind in ("pagerank", "eigenvector", "hubs", "authorities")
    for key in ("FollowProbability", "Tolerance", "MaxIterations"):
        if key in opts and (not iterative or (key == "FollowProbability" and kind != "pagerank")):
            raise MatlabError(f"centrality: the {key} option does not apply to '{kind}'.")
    tol = float(np.real(fnum(opts["Tolerance"])).flat[0]) if "Tolerance" in opts else 1e-4
    maxit = to_int(opts["MaxIterations"], "MaxIterations") if "MaxIterations" in opts else 100
    follow = float(np.real(fnum(opts["FollowProbability"])).flat[0]) if "FollowProbability" in opts else 0.85
    n = g.n
    w = np.ones(g.m) if importance is None else importance
    if kind in ("degree", "indegree", "outdegree"):
        out_d = np.bincount(g.src, weights=w, minlength=n) if g.m else np.zeros(n)
        in_d = np.bincount(g.dst, weights=w, minlength=n) if g.m else np.zeros(n)
        c = out_d + in_d if kind == "degree" else (in_d if kind == "indegree" else out_d)
    elif kind in ("closeness", "outcloseness", "incloseness"):
        c = _closeness(g, np.ones(g.m) if cost is None else cost, "bfs" if cost is None else "dijkstra",
                       reverse=(kind == "incloseness"))
    elif kind == "betweenness":
        c = _brandes(g, cost)
    elif kind == "pagerank":
        c = _pagerank(g, follow, tol, maxit, importance)
    else:
        a = np.zeros((n, n))
        np.add.at(a, (g.src, g.dst), w)
        if not g.directed:
            off = g.src != g.dst
            np.add.at(a, (g.dst[off], g.src[off]), w[off])
        if kind == "eigenvector":
            # the shift by the identity keeps the eigenvectors and makes the iteration converge
            c = _power_scores(a + np.eye(n), tol, maxit)
        elif kind == "hubs":
            c = _power_scores(a @ a.T, tol, maxit)
        else:
            c = _power_scores(a.T @ a, tol, maxit)
    return _col(c)


# -- structure of directed graphs -----------------------------------------------------------------------------

def _reach_matrix(g: GraphObject) -> np.ndarray:
    """reach[u, v] is true when a path with at least one edge leads from u to v."""
    n = g.n
    adj = g.out_lists()
    reach = np.zeros((n, n), dtype=bool)
    for s in range(n):
        stack = [v for v, _ in adj[s]]
        while stack:
            u = stack.pop()
            if not reach[s, u]:
                reach[s, u] = True
                stack.extend(v for v, _ in adj[u])
    return reach


@method("transclosure")
def _transclosure(interp, args, nargout):
    g = _g(args, "transclosure")
    if not g.directed:
        raise MatlabError("transclosure is only defined for directed graphs (digraph).")
    reach = _reach_matrix(g)
    src, dst = np.nonzero(reach)
    return GraphObject(True, g.n, src, dst, None, g.names, None, g.node_vars)


@method("transreduction")
def _transreduction(interp, args, nargout):
    g = _g(args, "transreduction")
    if not g.directed:
        raise MatlabError("transreduction is only defined for directed graphs (digraph).")
    comp = _strong_components(g)
    ncomp = int(comp.max()) + 1 if g.n else 0
    members = [np.nonzero(comp == c)[0] for c in range(ncomp)]
    # condensation, reduced: keep an edge between components only when no longer path exists
    cedges = {(int(comp[g.src[e]]), int(comp[g.dst[e]])) for e in range(g.m) if comp[g.src[e]] != comp[g.dst[e]]}
    cg = GraphObject(True, ncomp, [a for a, _ in sorted(cedges)], [b for _, b in sorted(cedges)])
    reach = _reach_matrix(cg)
    src: list[int] = []
    dst: list[int] = []
    for a, b in sorted(cedges):
        longer = any(reach[a, k] and reach[k, b] for k in range(ncomp) if k not in (a, b))
        if not longer:
            src.append(int(members[a][0]))
            dst.append(int(members[b][0]))
    for mem in members:  # each strong component becomes one simple cycle
        if mem.size > 1:
            for k in range(mem.size):
                src.append(int(mem[k]))
                dst.append(int(mem[(k + 1) % mem.size]))
    return GraphObject(True, g.n, src, dst, None, g.names, None, g.node_vars)


@method("condensation")
def _condensation(interp, args, nargout):
    g = _g(args, "condensation")
    if not g.directed:
        raise MatlabError("condensation is only defined for directed graphs (digraph).")
    comp = _strong_components(g)
    ncomp = int(comp.max()) + 1 if g.n else 0
    pairs = sorted({(int(comp[g.src[e]]), int(comp[g.dst[e]])) for e in range(g.m)
                    if comp[g.src[e]] != comp[g.dst[e]]})
    return GraphObject(True, ncomp, [a for a, _ in pairs], [b for _, b in pairs])


@method("simplify", public=False)
def _simplify_graph(interp, args, nargout):
    g = _g(args, "simplify")
    rest = list(args[1:])
    how = "min"
    keep_loops = False
    pick = None
    k = 0
    while k < len(rest):
        if not is_text(rest[k]):
            raise MatlabError("simplify: options must be text.")
        key = text(rest[k]).lower()
        if key in ("min", "max", "sum", "mean"):
            how = key
        elif key in ("keepselfloops", "omitselfloops"):
            keep_loops = key == "keepselfloops"
        elif key == "pickvariable" and k + 1 < len(rest):
            pick = text(rest[k + 1])
            k += 1
        elif key == "aggregationvariables" and k + 1 < len(rest):
            raise MatlabError("simplify: the AggregationVariables option is not supported.")
        else:
            raise MatlabError(f"simplify: unknown option '{text(rest[k])}'.")
        k += 1
    key_col = g.w
    if pick is not None:
        if pick == "Weight":
            key_col = g.w
        elif pick in g.edge_vars:
            key_col = np.real(fnum(g.edge_vars[pick])).astype(float).reshape(-1)
        else:
            raise MatlabError(f"simplify: the graph has no edge variable named '{pick}'.")
    groups: dict = {}
    for e in range(g.m):
        if g.src[e] == g.dst[e] and not keep_loops:
            continue
        groups.setdefault((int(g.src[e]), int(g.dst[e])), []).append(e)
    reps, weights = [], []
    for key in sorted(groups):
        es = groups[key]
        if key_col is None:
            reps.append(es[0])
            continue
        vals = key_col[es]
        if how == "min":
            reps.append(es[int(np.argmin(vals))])
            weights.append(float(vals.min()))
        elif how == "max":
            reps.append(es[int(np.argmax(vals))])
            weights.append(float(vals.max()))
        else:
            reps.append(es[0])
            weights.append(float(vals.sum() if how == "sum" else vals.mean()))
    idx = np.array(reps, dtype=np.int64)
    w = None
    if g.w is not None:
        w = np.array(weights) if (pick in (None, "Weight") and weights) else g.w[idx]
    h = GraphObject(g.directed, g.n, g.src[idx], g.dst[idx], w, g.names,
                    {k2: _take_rows(v, idx) for k2, v in g.edge_vars.items()}, g.node_vars, presorted=True)
    if nargout > 1:
        return h, _col(idx + 1.0)
    return h


@method("allpaths")
def _allpaths(interp, args, nargout):
    g = _g(args, "allpaths")
    if len(args) < 3:
        raise MatlabError("allpaths expects a graph, a source node and a target node.")
    s, named = g.one_node(args[1], "source node")
    t, _ = g.one_node(args[2], "target node")
    opts = _options(args[3:], {"maxnumpaths": "MaxNumPaths", "maxpathlength": "MaxPathLength",
                                "minpathlength": "MinPathLength"}, "allpaths")
    max_paths = float(np.real(fnum(opts["MaxNumPaths"])).flat[0]) if "MaxNumPaths" in opts else np.inf
    max_len = float(np.real(fnum(opts["MaxPathLength"])).flat[0]) if "MaxPathLength" in opts else np.inf
    min_len = float(np.real(fnum(opts["MinPathLength"])).flat[0]) if "MinPathLength" in opts else 0.0
    adj = g.out_lists()
    paths: list = []
    epaths: list = []
    on_path = np.zeros(g.n, dtype=bool)

    def walk(u, nodes, edges):
        for v, e in adj[u]:
            if len(paths) >= max_paths:
                return
            if v == t:
                if len(edges) + 1 >= min_len:
                    paths.append(nodes + [v])
                    epaths.append(edges + [e])
                continue
            if on_path[v] or len(edges) + 1 >= max_len:
                continue
            on_path[v] = True
            walk(v, nodes + [v], edges + [e])
            on_path[v] = False

    if s == t:
        if min_len <= 0:
            paths.append([s])
            epaths.append([])
    else:
        on_path[s] = True
        walk(s, [s], [])
    out = Cell.from_list([g.node_out(p, named, column=False) for p in paths], column=True)
    if nargout > 1:
        return out, Cell.from_list([_row([e + 1.0 for e in ep]) for ep in epaths], column=True)
    return out


# -- layouts ------------------------------------------------------------------------------------------------

def _layout_circle(n: int) -> np.ndarray:
    ang = 2 * np.pi * np.arange(n) / max(n, 1)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _layout_force(g: GraphObject, iterations: int = 100, dim: int = 2, weight_effect: str = "none",
                  start: np.ndarray | None = None, gravity: bool = False) -> np.ndarray:
    """Force-directed placement (Fruchterman and Reingold): edges attract, all node pairs repel."""
    n = g.n
    if n == 0:
        return np.zeros((0, dim))
    if n == 1:
        return np.zeros((1, dim))
    if start is not None:
        pos = np.array(start, dtype=float)
    else:
        # a deterministic start: points on a circle (or a spiral on a sphere), slightly staggered
        k = np.arange(n)
        ang = 2 * np.pi * k / n
        pos = np.column_stack([np.cos(ang), np.sin(ang)]) * (1 + 0.05 * np.sin(7.0 * k))[:, None]
        if dim == 3:
            pos = np.column_stack([pos, np.linspace(-1, 1, n)])
    off = g.src != g.dst
    src, dst = g.src[off], g.dst[off]
    strength = np.ones(src.size)
    if weight_effect != "none" and g.w is not None and src.size:
        w = np.abs(g.w[off])
        w = w / (w.mean() if w.mean() > 0 else 1.0)
        # 'direct': a heavier edge is longer (weaker spring); 'inverse': a heavier edge is shorter
        strength = 1.0 / np.maximum(w, 1e-6) if weight_effect == "direct" else w
    comp = _weak_components(g)
    pull = gravity or (int(comp.max()) > 0)
    ideal = 1.0
    temp = 0.5 * np.sqrt(n)
    cool = temp / (iterations + 1.0)
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(delta * delta, axis=2)
        np.fill_diagonal(d2, 1.0)
        disp = np.sum(delta * (ideal * ideal / d2)[:, :, None], axis=1)
        if src.size:
            ev = pos[src] - pos[dst]
            el = np.sqrt(np.maximum(np.sum(ev * ev, axis=1), 1e-12))
            f = (ev * (el * strength / ideal)[:, None])
            np.add.at(disp, src, -f)
            np.add.at(disp, dst, f)
        if pull:
            disp -= 0.3 * (pos - pos.mean(axis=0))
        length = np.sqrt(np.maximum(np.sum(disp * disp, axis=1), 1e-12))
        pos = pos + disp / length[:, None] * np.minimum(length, temp)[:, None]
        temp = max(temp - cool, 1e-3)
    return pos - pos.mean(axis=0)


def _layout_subspace(g: GraphObject, dim: int = 2) -> np.ndarray:
    """Spectral placement from the eigenvectors of the graph Laplacian with the smallest eigenvalues."""
    n = g.n
    if n <= dim:
        return _layout_circle(n) if dim == 2 else np.column_stack([_layout_circle(n), np.zeros(n)])
    a = np.zeros((n, n))
    off = g.src != g.dst
    a[g.src[off], g.dst[off]] = 1.0
    a[g.dst[off], g.src[off]] = 1.0
    lap = np.diag(a.sum(axis=1)) - a
    w, v = np.linalg.eigh(lap)
    ncomp = int(_weak_components(g).max()) + 1
    cols = v[:, ncomp:ncomp + dim] if ncomp + dim <= n else v[:, -dim:]
    cols = cols * np.where(cols[np.argmax(np.abs(cols), axis=0), np.arange(cols.shape[1])] < 0, -1.0, 1.0)
    return cols * np.sqrt(n)


def _layout_layered(g: GraphObject, direction: str = "down", sources=None, sinks=None,
                    assign: str = "auto") -> np.ndarray:
    """Layered drawing: nodes in rows by rank, ordered inside each row to shorten the edges."""
    n = g.n
    if n == 0:
        return np.zeros((0, 2))
    # orient every edge along a depth-first order so that the layering sees no cycle
    adj = g.out_lists()
    state = np.zeros(n, dtype=np.int8)
    fwd: list[tuple[int, int]] = []
    seen_edge = np.zeros(g.m, dtype=bool)
    roots = list(sources) if sources is not None else []
    indeg = np.bincount(g.dst, minlength=n) if (g.m and g.directed) else np.zeros(n, dtype=np.int64)
    roots += [v for v in range(n) if indeg[v] == 0 and v not in roots] + [v for v in range(n)]
    for root in roots:
        if state[root]:
            continue
        state[root] = 1
        stack = [(root, 0)]
        while stack:
            u, k = stack[-1]
            if k < len(adj[u]):
                stack[-1] = (u, k + 1)
                v, e = adj[u][k]
                if seen_edge[e] or u == v:
                    continue
                seen_edge[e] = True
                if state[v] == 1:
                    fwd.append((v, u))  # a back edge: drawn against its direction
                else:
                    fwd.append((u, v))
                    if state[v] == 0:
                        state[v] = 1
                        stack.append((v, 0))
            else:
                stack.pop()
                state[u] = 2
    succ: list = [[] for _ in range(n)]
    pred: list = [[] for _ in range(n)]
    for u, v in fwd:
        succ[u].append(v)
        pred[v].append(u)
    # longest-path ranks
    order = []
    deg = np.array([len(p) for p in pred])
    ready = [v for v in range(n) if deg[v] == 0]
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in succ[u]:
            deg[v] -= 1
            if deg[v] == 0:
                ready.append(v)
    rank = np.zeros(n, dtype=np.int64)
    for u in order:
        for v in succ[u]:
            rank[v] = max(rank[v], rank[u] + 1)
    top = int(rank.max()) if n else 0
    if assign == "alap":
        rank = np.full(n, top, dtype=np.int64)
        for u in reversed(order):
            for v in succ[u]:
                rank[u] = min(rank[u], rank[v] - 1)
    if sinks is not None:
        for v in sinks:
            rank[v] = top
    if sources is not None:
        for v in sources:
            rank[v] = 0
    layers = [[v for v in range(n) if rank[v] == r] for r in range(top + 1)]
    place = np.zeros(n)
    for layer in layers:
        for k, v in enumerate(layer):
            place[v] = k
    for sweep in range(8):  # barycentre ordering, alternately downwards and upwards
        rng = range(1, top + 1) if sweep % 2 == 0 else range(top - 1, -1, -1)
        nb = pred if sweep % 2 == 0 else succ
        for r in rng:
            layer = layers[r]
            key = {v: (np.mean([place[u] for u in nb[v]]) if nb[v] else place[v]) for v in layer}
            layer.sort(key=lambda v: (key[v], v))
            for k, v in enumerate(layer):
                place[v] = k
    xy = np.zeros((n, 2))
    width = max(len(layer) for layer in layers)
    for r, layer in enumerate(layers):
        shift = (width - len(layer)) / 2.0
        for k, v in enumerate(layer):
            xy[v] = (1.0 + k + shift, float(top + 1 - r))
    if direction == "up":
        xy[:, 1] = top + 2 - xy[:, 1]
    elif direction in ("left", "right"):
        xy = xy[:, ::-1].copy()
        if direction == "right":
            xy[:, 0] = top + 2 - xy[:, 0]
    return xy


_LAYOUTS = ("auto", "circle", "force", "layered", "subspace", "force3", "subspace3")


def _compute_layout(g: GraphObject, method: str, opts: dict, name: str) -> np.ndarray:
    allowed = {
        "circle": {"center"},
        "force": {"iterations", "weighteffect", "usegravity", "xstart", "ystart"},
        "force3": {"iterations", "weighteffect", "usegravity", "xstart", "ystart", "zstart"},
        "layered": {"direction", "sources", "sinks", "assignlayers"},
        "subspace": {"dimension"},
        "subspace3": {"dimension"},
    }
    if method == "auto":
        if opts:
            raise MatlabError(f"{name}: layout options need an explicit layout method.")
        if g.directed and g.n and _is_dag(g):
            method = "layered"
        elif g.n > 100:
            method = "subspace"
        else:
            method = "force"
    for key in opts:
        if key not in allowed[method]:
            raise MatlabError(f"{name}: option '{key}' does not apply to the '{method}' layout.")
    if method == "circle":
        xy = _layout_circle(g.n)
        if "center" in opts:
            c = g.one_node(opts["center"], "center node")[0]
            others = [v for v in range(g.n) if v != c]
            xy = np.zeros((g.n, 2))
            xy[others] = _layout_circle(len(others))
        return xy
    if method in ("force", "force3"):
        dim = 3 if method == "force3" else 2
        iters = to_int(opts["iterations"], "Iterations") if "iterations" in opts else 100
        effect = _choice(opts["weighteffect"], ("none", "direct", "inverse"), name, "WeightEffect") \
            if "weighteffect" in opts else "none"
        if effect != "none" and g.w is None:
            raise MatlabError(f"{name}: WeightEffect needs a graph with edge weights.")
        grav = bool(np.all(np.asarray(fnum(opts["usegravity"])) != 0)) if "usegravity" in opts else False
        start = None
        keys = ["xstart", "ystart"] + (["zstart"] if dim == 3 else [])
        if any(k in opts for k in keys):
            if not all(k in opts for k in keys):
                raise MatlabError(f"{name}: give all of XStart, YStart" + (", ZStart" if dim == 3 else "") + ".")
            cols = [np.real(fnum(opts[k])).astype(float).reshape(-1) for k in keys]
            if any(c.size != g.n for c in cols):
                raise MatlabError(f"{name}: the start coordinates need one element per node.")
            start = np.column_stack(cols)
        return _layout_force(g, iters, dim, effect, start, grav)
    if method in ("subspace", "subspace3"):
        if "dimension" in opts:
            to_int(opts["dimension"], "Dimension")  # the embedding dimension only tunes the projection
        return _layout_subspace(g, 3 if method == "subspace3" else 2)
    direction = _choice(opts["direction"], ("down", "up", "left", "right"), name, "Direction") \
        if "direction" in opts else "down"
    assign = _choice(opts["assignlayers"], ("auto", "asap", "alap"), name, "AssignLayers") \
        if "assignlayers" in opts else "auto"
    srcs = [int(v) for v in g.node_ids(opts["sources"])[0]] if "sources" in opts else None
    snks = [int(v) for v in g.node_ids(opts["sinks"])[0]] if "sinks" in opts else None
    return _layout_layered(g, direction, srcs, snks, assign)


# -- drawing --------------------------------------------------------------------------------------------------

_COLOR_CODES = {"r": (1, 0, 0), "g": (0, 1, 0), "b": (0, 0, 1), "c": (0, 1, 1), "m": (1, 0, 1),
                "y": (1, 1, 0), "k": (0, 0, 0), "w": (1, 1, 1), "red": (1, 0, 0), "green": (0, 1, 0),
                "blue": (0, 0, 1), "cyan": (0, 1, 1), "magenta": (1, 0, 1), "yellow": (1, 1, 0),
                "black": (0, 0, 0), "white": (1, 1, 1)}
_DEFAULT_BLUE = (0.0, 0.447, 0.741)


def _one_color(v: Any, what: str) -> tuple | None:
    if is_text(v):
        s = text(v).lower()
        if s == "none":
            return None
        if s in _COLOR_CODES:
            return tuple(float(c) for c in _COLOR_CODES[s])
        if s.startswith("#") and len(s) in (4, 7):
            h = s[1:] if len(s) == 7 else "".join(ch * 2 for ch in s[1:])
            try:
                return tuple(int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
            except ValueError:
                pass
        raise MatlabError(f"plot: '{text(v)}' is not a valid {what}.")
    a = np.real(fnum(v)).astype(float)
    if a.size != 3 or np.any(a < 0) or np.any(a > 1):
        raise MatlabError(f"plot: a {what} must be a color name or an RGB triple with values in [0, 1].")
    return tuple(float(c) for c in a.reshape(-1))


def _color_list(v: Any, count: int, what: str) -> list:
    """One color per node or edge from a single color or a count-by-3 matrix."""
    if isinstance(v, np.ndarray) and not is_char(v) and v.ndim == 2 and v.shape == (count, 3) and count != 1:
        a = np.real(fnum(v)).astype(float)
        return [tuple(float(c) for c in row) for row in a]
    return [_one_color(v, what)] * count


def _cdata_colors(interp, values: np.ndarray) -> list:
    cmap = np.asarray(BUILTINS["parula"](interp, [np.array([[256.0]])], 1), dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = hi - lo if hi > lo else 1.0
    idx = np.clip(((values - lo) / span * 255).astype(int), 0, 255)
    return [tuple(float(c) for c in cmap[i]) for i in idx]


_MARKERS = {"o", "+", "*", ".", "x", "_", "|", "s", "d", "^", "v", ">", "<", "p", "h", "none",
            "square", "diamond", "pentagram", "hexagram"}
_LINESTYLES = {"-", "--", ":", "-.", "none"}


class GraphPlotObject(MObject):
    """The handle returned by plot(G): node positions and styles, with highlight and layout methods."""

    is_handle = True

    def __init__(self, g: GraphObject, xy: np.ndarray, target: list):
        self.g = g
        self.xy = xy
        self.target = target  # an axes handle given to plot, if any
        n, m = g.n, g.m
        self.node_color: list = [_DEFAULT_BLUE] * n
        self.edge_color: list = [_DEFAULT_BLUE] * m
        self.marker: list = ["o"] * n
        self.marker_size = np.full(n, 4.0)
        self.line_width = np.full(m, 0.5)
        self.line_style: list = ["-"] * m
        self.node_label: list[str] = []
        self.edge_label: list[str] = []
        self.node_font_size = 8.0
        self.edge_font_size = 8.0
        self.arrow_size = 7.0
        self.show_arrows = g.directed
        self.edge_alpha = 0.5
        self.layout_name = "force"

    def class_name(self) -> str:
        return "matlab.graphics.chart.primitive.GraphPlot"

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "GraphPlot", "handle")

    _METHOD_NAMES = {"highlight", "layout", "labelnode", "labeledge"}

    def has_method(self, name: str) -> bool:
        return name in self._METHOD_NAMES

    # -- properties -------------------------------------------------------------------------------
    def _label_value(self, labels: list[str]):
        return Cell.from_list([char(s) for s in labels]) if labels else Cell(np.empty((1, 0), dtype=object))

    def get_field(self, interp, name: str) -> Any:
        key = name.lower()
        if key == "xdata":
            return _row(self.xy[:, 0])
        if key == "ydata":
            return _row(self.xy[:, 1])
        if key == "zdata":
            return _row(self.xy[:, 2]) if self.xy.shape[1] > 2 else np.zeros((1, self.g.n))
        if key == "nodelabel":
            return self._label_value(self.node_label)
        if key == "edgelabel":
            return self._label_value(self.edge_label)
        if key == "nodecolor":
            return np.array([c if c is not None else (np.nan,) * 3 for c in self.node_color]).reshape(-1, 3) \
                if len(set(self.node_color)) > 1 else np.array([self.node_color[0]]) if self.node_color \
                else np.zeros((0, 3))
        if key == "edgecolor":
            return np.array([c if c is not None else (np.nan,) * 3 for c in self.edge_color]).reshape(-1, 3) \
                if len(set(self.edge_color)) > 1 else np.array([self.edge_color[0]]) if self.edge_color \
                else np.zeros((0, 3))
        if key == "markersize":
            return _row(self.marker_size) if np.unique(self.marker_size).size > 1 else \
                (float(self.marker_size[0]) if self.marker_size.size else 4.0)
        if key == "linewidth":
            return _row(self.line_width) if np.unique(self.line_width).size > 1 else \
                (float(self.line_width[0]) if self.line_width.size else 0.5)
        if key == "marker":
            return char(self.marker[0] if self.marker else "o")
        if key == "linestyle":
            return char(self.line_style[0] if self.line_style else "-")
        if key == "nodefontsize":
            return self.node_font_size
        if key == "edgefontsize":
            return self.edge_font_size
        if key == "arrowsize":
            return self.arrow_size
        if key == "showarrows":
            return char("on" if self.show_arrows else "off")
        if key == "edgealpha":
            return self.edge_alpha
        if key == "type":
            return char("graphplot")
        raise MatlabError(f"Unrecognized property '{name}' for class 'GraphPlot'.")

    def apply(self, interp, name: str, value: Any) -> None:
        """Set one property (without redrawing)."""
        g = self.g
        key = name.lower()

        def per(count, what, lo=0.0):
            a = np.real(fnum(value)).astype(float).reshape(-1)
            if a.size == 1:
                a = np.full(count, a[0])
            if a.size != count or np.any(a <= lo):
                raise MatlabError(f"plot: {name} must be a positive scalar or one value per {what}.")
            return a

        def labels(count, what):
            if isinstance(value, np.ndarray) and value.size == 0:
                return []
            if isinstance(value, np.ndarray) and not is_char(value):
                vals = np.real(fnum(value)).reshape(-1, order="F")
                out = [("%g" % v) for v in vals]
            else:
                out = _name_list(value)
            if len(out) != count:
                raise MatlabError(f"plot: {name} needs one label per {what}.")
            return out

        if key in ("xdata", "ydata", "zdata"):
            a = np.real(fnum(value)).astype(float).reshape(-1)
            if a.size != g.n:
                raise MatlabError(f"plot: {name} needs one coordinate per node.")
            col = "xyz".index(key[0])
            if col == 2 and self.xy.shape[1] == 2:
                self.xy = np.column_stack([self.xy, np.zeros(g.n)])
            self.xy[:, col] = a
        elif key == "nodelabel":
            self.node_label = labels(g.n, "node")
        elif key == "edgelabel":
            self.edge_label = labels(g.m, "edge")
        elif key == "nodecolor":
            self.node_color = _color_list(value, g.n, "node color")
        elif key == "edgecolor":
            self.edge_color = _color_list(value, g.m, "edge color")
        elif key == "nodecdata":
            a = np.real(fnum(value)).astype(float).reshape(-1)
            if a.size != g.n:
                raise MatlabError("plot: NodeCData needs one value per node.")
            self.node_color = _cdata_colors(interp, a)
        elif key == "edgecdata":
            a = np.real(fnum(value)).astype(float).reshape(-1)
            if a.size != g.m:
                raise MatlabError("plot: EdgeCData needs one value per edge.")
            self.edge_color = _cdata_colors(interp, a)
        elif key == "markersize":
            self.marker_size = per(g.n, "node")
        elif key == "linewidth":
            self.line_width = per(g.m, "edge")
        elif key == "marker":
            vals = _name_list(value)
            if any(v.lower() not in _MARKERS for v in vals) or len(vals) not in (1, g.n):
                raise MatlabError("plot: Marker must be a marker symbol, or one per node.")
            self.marker = [v.lower() for v in (vals * g.n if len(vals) == 1 else vals)]
        elif key == "linestyle":
            vals = _name_list(value)
            if any(v.lower() not in _LINESTYLES for v in vals) or len(vals) not in (1, g.m):
                raise MatlabError("plot: LineStyle must be '-', '--', ':', '-.' or 'none', or one per edge.")
            self.line_style = [v.lower() for v in (vals * g.m if len(vals) == 1 else vals)]
        elif key == "nodefontsize":
            self.node_font_size = float(per(1, "plot")[0])
        elif key == "edgefontsize":
            self.edge_font_size = float(per(1, "plot")[0])
        elif key == "arrowsize":
            self.arrow_size = float(per(1, "plot")[0])
        elif key == "edgealpha":
            a = float(np.real(fnum(value)).flat[0])
            if not 0 <= a <= 1:
                raise MatlabError("plot: EdgeAlpha must be between 0 and 1.")
            self.edge_alpha = a
        elif key == "showarrows":
            if is_text(value):
                self.show_arrows = _choice(value, ("on", "off"), "plot", "ShowArrows") == "on"
            else:
                self.show_arrows = bool(np.all(np.asarray(fnum(value)) != 0))
            if self.show_arrows and not g.directed:
                raise MatlabError("plot: arrows can only be shown for a directed graph.")
        else:
            raise MatlabError(f"Unrecognized property '{name}' for class 'GraphPlot'.")

    def set_field(self, interp, name: str, value: Any) -> "GraphPlotObject":
        self.apply(interp, name, value)
        self.draw(interp, clear=True)
        return self

    # -- drawing ----------------------------------------------------------------------------------
    def _call(self, interp, fname: str, cargs: list) -> None:
        interp.call_named(fname, [*self.target, *cargs], 0, interp.frame)

    def draw(self, interp, clear: bool) -> None:
        g, xy = self.g, self.xy
        three_d = xy.shape[1] > 2
        frame = interp.frame
        held = bool(np.all(np.asarray(interp.call_named("ishold", list(self.target), 1, frame)[0]) != 0))
        if clear:
            interp.call_named("cla", list(self.target), 0, frame)
        elif not held:
            interp.call_named("cla", list(self.target), 0, frame)
        interp.call_named("hold", [*self.target, char("on")], 0, frame)
        try:
            span = float(np.max(xy.max(axis=0) - xy.min(axis=0))) if g.n else 1.0
            span = span if span > 0 else 1.0
            # edges, grouped by style
            groups: dict = {}
            for e in range(g.m):
                if self.edge_color[e] is None or self.line_style[e] == "none":
                    continue
                groups.setdefault((self.edge_color[e], float(self.line_width[e]), self.line_style[e]), []).append(e)
            for (color, width, style), edges in groups.items():
                cols: list = [[] for _ in range(xy.shape[1])]
                for e in edges:
                    a, b = xy[g.src[e]], xy[g.dst[e]]
                    if g.src[e] == g.dst[e]:
                        # a self-loop: a small circle beside the node
                        t = np.linspace(0, 2 * np.pi, 25)
                        r = 0.04 * span
                        pts = [a[0] + r * (1 - np.cos(t)), a[1] + r * np.sin(t)]
                        if three_d:
                            pts.append(np.full(t.size, a[2]))
                    else:
                        pts = [np.array([a[k], b[k]]) for k in range(xy.shape[1])]
                    for k in range(xy.shape[1]):
                        cols[k].extend(list(pts[k]) + [np.nan])
                data = [_row(c) for c in cols]
                self._call(interp, "plot3" if three_d else "plot",
                           [*data, char(style), char("Color"), _row(color), char("LineWidth"), width])
            # arrow heads for directed graphs
            if self.show_arrows and g.directed and g.m and not three_d:
                # arrow heads are shaped in screen units (a 4:3 axes box is assumed), then mapped back
                ext = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-12) * 1.16 if g.n else np.ones(2)
                scale = np.array([4.0 / ext[0], 3.0 / ext[1]])
                size = 0.045 * self.arrow_size / 7.0
                by_color: dict = {}
                for e in range(g.m):
                    if g.src[e] == g.dst[e] or self.edge_color[e] is None:
                        continue
                    a, b = xy[g.src[e]] * scale, xy[g.dst[e]] * scale
                    d = b - a
                    length = float(np.hypot(d[0], d[1]))
                    if length == 0:
                        continue
                    u = d / length
                    perp = np.array([-u[1], u[0]])
                    tip = a + 0.5 * d + u * size
                    base = tip - 2 * size * u
                    tri = np.array([tip, base + 0.6 * size * perp, base - 0.6 * size * perp]) / scale
                    by_color.setdefault(self.edge_color[e], []).append(tri)
                for color, tris in by_color.items():
                    xs = np.column_stack([t[:, 0] for t in tris])
                    ys = np.column_stack([t[:, 1] for t in tris])
                    if len(tris) == 1:
                        xs, ys = xs.reshape(1, -1), ys.reshape(1, -1)
                    self._call(interp, "patch", [xs, ys, _row(color), char("EdgeColor"), char("none")])
            # nodes, grouped by style
            ngroups: dict = {}
            for v in range(g.n):
                if self.node_color[v] is None or self.marker[v] == "none":
                    continue
                ngroups.setdefault((self.node_color[v], float(self.marker_size[v]), self.marker[v]), []).append(v)
            for (color, size, marker), nodes in ngroups.items():
                data = [_row(xy[nodes, k]) for k in range(xy.shape[1])]
                self._call(interp, "plot3" if three_d else "plot",
                           [*data, char("LineStyle"), char("none"), char("Marker"), char(marker),
                            char("MarkerSize"), size, char("MarkerFaceColor"), _row(color),
                            char("MarkerEdgeColor"), _row(color)])
            off = 0.025 * span
            for v, label in enumerate(self.node_label):
                if label == "":
                    continue
                pos = [float(xy[v, 0] + off), float(xy[v, 1] - off)] + ([float(xy[v, 2])] if three_d else [])
                self._call(interp, "text", [*pos, char(label), char("FontSize"), self.node_font_size])
            for e, label in enumerate(self.edge_label):
                if label == "":
                    continue
                mid = (xy[g.src[e]] + xy[g.dst[e]]) / 2.0
                pos = [float(mid[0]), float(mid[1])] + ([float(mid[2])] if three_d else [])
                self._call(interp, "text", [*pos, char(label), char("FontSize"), self.edge_font_size])
            if g.n and not three_d:
                pad = 0.08 * span + 1e-9
                lo, hi = xy.min(axis=0), xy.max(axis=0)
                for fname, k in (("xlim", 0), ("ylim", 1)):
                    width = hi[k] - lo[k]
                    extra = pad if width > 0 else 0.5
                    self._call(interp, fname, [_row([lo[k] - extra, hi[k] + extra])])
                try:
                    ax = self.target[0] if self.target else interp.call_named("gca", [], 1, frame)[0]
                    interp.call_named("set", [ax, char("XTick"), np.zeros((1, 0)), char("YTick"),
                                              np.zeros((1, 0))], 0, frame)
                except MatlabError:
                    pass
        finally:
            if not held:
                interp.call_named("hold", [*self.target, char("off")], 0, frame)

    # -- methods ----------------------------------------------------------------------------------
    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        g = self.g
        rest = [a for a in args if a is not self]
        if name == "layout":
            method = "auto"
            if rest and is_text(rest[0]) and text(rest[0]).lower() in _LAYOUTS:
                method = text(rest.pop(0)).lower()
            if len(rest) % 2:
                raise MatlabError("layout: options must be given as name-value pairs.")
            opts = {text(rest[k]).lower(): rest[k + 1] for k in range(0, len(rest), 2)}
            self.xy = _compute_layout(g, method, opts, "layout")
            self.layout_name = method
            self.draw(interp, clear=True)
            return []
        if name in ("labelnode", "labeledge"):
            if name == "labelnode":
                if len(rest) != 2:
                    raise MatlabError("labelnode expects (H, nodeIDs, labels).")
                ids = g.node_ids(rest[0])[0]
                store, count = self.node_label, g.n
            else:
                if len(rest) == 3:
                    s, t, _ = _pairs(g, rest[0], rest[1], "labeledge")
                    table = _edge_lookup(g)
                    ids = np.array([e for k in range(s.size) for e in table.get(
                        (int(s[k]), int(t[k])) if g.directed else (min(int(s[k]), int(t[k])),
                                                                   max(int(s[k]), int(t[k]))), [])],
                        dtype=np.int64)
                    rest = [None, rest[2]]
                elif len(rest) == 2:
                    ids = np.real(fnum(rest[0])).reshape(-1).astype(np.int64) - 1
                else:
                    raise MatlabError("labeledge expects (H, edgeIndices, labels) or (H, s, t, labels).")
                store, count = self.edge_label, g.m
            lab = rest[1]
            if isinstance(lab, np.ndarray) and not is_char(lab):
                texts = [("%g" % v) for v in np.real(fnum(lab)).reshape(-1, order="F")]
            else:
                texts = _name_list(lab)
            if len(texts) == 1 and ids.size > 1:
                texts = texts * ids.size
            if len(texts) != ids.size:
                raise MatlabError(f"{name}: give one label, or one label per element.")
            if not store:
                store.extend([""] * count)
            for i, s in zip(ids, texts):
                store[int(i)] = s
            self.draw(interp, clear=True)
            return []
        if name == "highlight":
            self._highlight(interp, rest)
            return []
        return super().call_method(interp, name, args, nargout)

    def _highlight(self, interp, rest: list) -> None:
        g = self.g
        style_names = {"nodecolor", "edgecolor", "markersize", "linewidth", "marker", "linestyle",
                       "nodefontsize", "edgefontsize", "arrowsize"}
        pos: list = []
        while rest and not (is_text(rest[0]) and not isinstance(rest[0], Cell)
                            and text(rest[0]).lower() in style_names | {"edges"} and len(rest) % 2 == 0):
            pos.append(rest.pop(0))
        edge_sel = None
        if rest and text(rest[0]).lower() == "edges":
            edge_sel = np.real(fnum(rest[1])).reshape(-1).astype(np.int64) - 1
            rest = rest[2:]
        opts = {text(rest[k]).lower(): rest[k + 1] for k in range(0, len(rest), 2)}
        for key in opts:
            if key not in style_names:
                raise MatlabError(f"highlight: unknown option '{key}'.")
        nodes = np.zeros(0, dtype=np.int64)
        edges = np.zeros(0, dtype=np.int64) if edge_sel is None else edge_sel
        if len(pos) == 1 and isinstance(pos[0], GraphObject):
            sub = pos[0]
            if sub.n != g.n:
                raise MatlabError("highlight: the graph to highlight must have the same nodes as the plotted one.")
            table = _edge_lookup(g)
            hits = [e for k in range(sub.m) for e in table.get((int(sub.src[k]), int(sub.dst[k])), [])[:1]]
            edges = np.array(hits, dtype=np.int64)
            nodes = np.unique(np.concatenate([sub.src, sub.dst])) if sub.m else nodes
        elif len(pos) == 1:
            nodes = g.node_ids(pos[0])[0]
            if any(k in opts for k in ("edgecolor", "linewidth", "linestyle")):
                # a node list given with edge properties is a path: its consecutive nodes select edges
                table = _edge_lookup(g)
                hits = []
                for a, b in zip(nodes[:-1], nodes[1:]):
                    a, b = int(a), int(b)
                    hits.extend(table.get((a, b) if g.directed else (min(a, b), max(a, b)), [])[:1])
                edges = np.array(hits, dtype=np.int64)
        elif len(pos) == 2:
            s, t, _ = _pairs(g, pos[0], pos[1], "highlight")
            table = _edge_lookup(g)
            hits = []
            for k in range(s.size):
                a, b = int(s[k]), int(t[k])
                hits.extend(table.get((a, b) if g.directed else (min(a, b), max(a, b)), []))
            edges = np.array(hits, dtype=np.int64)
        elif pos:
            raise MatlabError("highlight expects (H, nodeIDs), (H, G), (H, s, t) or (H, 'Edges', idx).")
        if not opts:
            # the default emphasis: larger markers and thicker lines
            self.marker_size[nodes] = self.marker_size[nodes] * 1.75
            self.line_width[edges] = self.line_width[edges] * 3.0
        for key, val in opts.items():
            if key == "nodecolor":
                c = _one_color(val, "node color")
                for v in nodes:
                    self.node_color[int(v)] = c
            elif key == "edgecolor":
                c = _one_color(val, "edge color")
                for e in edges:
                    self.edge_color[int(e)] = c
            elif key == "markersize":
                self.marker_size[nodes] = float(np.real(fnum(val)).flat[0])
            elif key == "linewidth":
                self.line_width[edges] = float(np.real(fnum(val)).flat[0])
            elif key == "marker":
                for v in nodes:
                    self.marker[int(v)] = text(val).lower()
            elif key == "linestyle":
                for e in edges:
                    self.line_style[int(e)] = text(val).lower()
            else:
                self.apply(interp, key, val)
        self.draw(interp, clear=True)

    def display_lines(self) -> list[str]:
        return ["  GraphPlot with properties:", "",
                f"     NodeColor: [{' '.join('%g' % c for c in (self.node_color[0] or (0, 0, 0)))}]"
                if self.node_color else "     NodeColor: []",
                f"    MarkerSize: {self.marker_size[0] if self.marker_size.size else 4:g}",
                f"     LineWidth: {self.line_width[0] if self.line_width.size else 0.5:g}",
                f"         XData: [1×{self.g.n} double]",
                f"         YData: [1×{self.g.n} double]"]


_PLOT_PROPS = {"layout", "xdata", "ydata", "zdata", "nodelabel", "edgelabel", "nodecolor", "edgecolor",
               "nodecdata", "edgecdata", "markersize", "linewidth", "marker", "linestyle", "nodefontsize",
               "edgefontsize", "arrowsize", "edgealpha", "showarrows"}
_LAYOUT_OPTS = {"center", "iterations", "weighteffect", "usegravity", "xstart", "ystart", "zstart",
                "direction", "sources", "sinks", "assignlayers", "dimension"}


@method("plot", public=False)
def _plot_graph(interp, args, nargout):
    args = list(args)
    target: list = []
    while args and not isinstance(args[0], GraphObject):
        target.append(args.pop(0))  # an axes handle given first
    if not args:
        raise MatlabError("plot: expected a graph or digraph object.")
    if len(target) > 1:
        raise MatlabError("plot: unexpected arguments before the graph.")
    g = args.pop(0)
    style: dict = {}
    if len(args) % 2 == 1:
        # a line specification such as '-.dr'
        spec = text(args.pop(0)) if is_text(args[0]) else None
        if spec is None:
            raise MatlabError("plot: options after the graph must be a line specification and name-value pairs.")
        rest_spec = spec
        for ls in ("--", "-.", ":", "-"):
            if ls in rest_spec:
                style["linestyle"] = char(ls)
                rest_spec = rest_spec.replace(ls, "", 1)
                break
        for ch in rest_spec:
            if ch in _COLOR_CODES:
                style["nodecolor"] = char(ch)
                style["edgecolor"] = char(ch)
            elif ch in "o+*.xsd^v><ph":
                style["marker"] = char(ch)
            else:
                raise MatlabError(f"plot: '{spec}' is not a valid line specification.")
    opts: dict = {}
    for k in range(0, len(args), 2):
        if not is_text(args[k]):
            raise MatlabError("plot: option names must be text.")
        key = text(args[k]).lower()
        if key not in _PLOT_PROPS and key not in _LAYOUT_OPTS:
            raise MatlabError(f"plot: unknown graph plot option '{text(args[k])}'.")
        opts[key] = args[k + 1]
    layout = "auto"
    if "layout" in opts:
        layout = _choice(opts.pop("layout"), _LAYOUTS, "plot", "Layout")
    layout_opts = {k: opts.pop(k) for k in list(opts) if k in _LAYOUT_OPTS}
    given_xy = "xdata" in opts or "ydata" in opts
    if given_xy:
        if not ("xdata" in opts and "ydata" in opts):
            raise MatlabError("plot: give both XData and YData.")
        if layout != "auto" or layout_opts:
            raise MatlabError("plot: XData/YData cannot be combined with a Layout.")
        xy = np.zeros((g.n, 2))
    else:
        xy = _compute_layout(g, layout, layout_opts, "plot")
    h = GraphPlotObject(g, xy, target)
    h.layout_name = layout
    # default labels: node names or numbers, for graphs that are small enough to read
    if g.n <= 100:
        h.node_label = list(g.names) if g.names is not None else [str(v + 1) for v in range(g.n)]
    for key, val in {**style, **opts}.items():
        h.apply(interp, key, val)
    h.draw(interp, clear=False)
    return h if nargout > 0 else None


@builtin("gplot")
def _gplot(interp, args, nargout):
    if len(args) < 2 or len(args) > 3:
        raise MatlabError("gplot expects an adjacency matrix, node coordinates and an optional line style.")
    a = _adjacency_arg(args[0], "gplot").tocoo()
    xy = np.real(fnum(args[1])).astype(float)
    n = a.shape[0]
    if xy.ndim != 2 or xy.shape[0] != n or xy.shape[1] < 2:
        raise MatlabError("gplot: the coordinates must have one row per node and at least two columns.")
    xs: list = []
    ys: list = []
    for i, j in zip(a.row, a.col):
        xs.extend([xy[i, 0], xy[j, 0], np.nan])
        ys.extend([xy[i, 1], xy[j, 1], np.nan])
    if nargout >= 2:
        return _col(xs), _col(ys)
    spec = [args[2]] if len(args) > 2 else []
    res = interp.call_named("plot", [_row(xs), _row(ys), *spec], 1 if nargout else 0, interp.frame)
    return res[0] if nargout and res else None


# -- structure of sparse matrices -----------------------------------------------------------------------------

def _pattern(v: Any, name: str):
    import scipy.sparse as sp
    if hasattr(v, "to_dense") and hasattr(v, "m"):
        return sp.csc_matrix(v.m)
    a = fnum(v)
    if a.ndim != 2:
        raise MatlabError(f"{name}: the input must be a 2-D matrix.")
    return sp.csc_matrix(a)


def _matching(m) -> np.ndarray:
    """Maximum matching of rows to columns: for every column the matched row, or -1."""
    from scipy.sparse.csgraph import maximum_bipartite_matching
    return np.asarray(maximum_bipartite_matching((m != 0).tocsr(), perm_type="row"), dtype=np.int64)


@builtin("dmperm")
def _dmperm(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("dmperm expects one matrix.")
    if nargout > 6:
        raise MatlabError("dmperm: too many output arguments.")
    a = _pattern(args[0], "dmperm")
    m, n = a.shape
    pat = (a != 0).tocsr()
    match_row = _matching(a)             # match_row[j]: row matched to column j, or -1
    if nargout <= 1:
        # p(j) = i when column j is matched to row i, else 0
        return _row(match_row + 1.0)
    row_match = -np.ones(m, dtype=np.int64)
    for j, i in enumerate(match_row):
        if i >= 0:
            row_match[i] = j
    csc = pat.tocsc()
    # coarse decomposition by alternating paths from the unmatched rows and columns
    row_h = np.zeros(m, dtype=bool)   # rows reachable from unmatched columns (horizontal part)
    col_h = np.zeros(n, dtype=bool)
    stack = [j for j in range(n) if match_row[j] < 0]
    for j in stack:
        col_h[j] = True
    while stack:
        j = stack.pop()
        for i in csc.indices[csc.indptr[j]:csc.indptr[j + 1]]:
            if not row_h[i]:
                row_h[i] = True
                j2 = row_match[i]
                if j2 >= 0 and not col_h[j2]:
                    col_h[j2] = True
                    stack.append(j2)
    row_v = np.zeros(m, dtype=bool)   # rows reachable from unmatched rows (vertical part)
    col_v = np.zeros(n, dtype=bool)
    stack = [i for i in range(m) if row_match[i] < 0]
    for i in stack:
        row_v[i] = True
    while stack:
        i = stack.pop()
        for j in pat.indices[pat.indptr[i]:pat.indptr[i + 1]]:
            if not col_v[j]:
                col_v[j] = True
                i2 = match_row[j]
                if i2 >= 0 and not row_v[i2]:
                    row_v[i2] = True
                    stack.append(i2)
    cols_h = [j for j in range(n) if col_h[j]]
    cols_s = [j for j in range(n) if not col_h[j] and not col_v[j]]
    rows_s = [int(match_row[j]) for j in cols_s]
    cols_v = [j for j in range(n) if col_v[j]]
    rows_v = [i for i in range(m) if row_v[i]]
    # horizontal part: matched columns first so that its diagonal is nonzero; unmatched columns lead
    h_un = [j for j in cols_h if match_row[j] < 0]
    h_ma = [j for j in cols_h if match_row[j] >= 0]
    rows_h_order = [int(match_row[j]) for j in h_ma]
    # vertical part: matched rows first, the unmatched rows go last
    v_ma = [j for j in cols_v]
    rows_v_order = [int(match_row[j]) for j in v_ma] + [i for i in rows_v if row_match[i] < 0]
    # fine decomposition of the square part into strongly connected blocks
    ns = len(cols_s)
    blocks_p: list[int] = []
    blocks_q: list[int] = []
    bounds_r = [len(rows_h_order)]
    bounds_s = [len(h_un) + len(h_ma)]
    if ns:
        pos = {j: k for k, j in enumerate(cols_s)}
        src, dst = [], []
        for k in range(ns):
            i = rows_s[k]
            for j2 in pat.indices[pat.indptr[i]:pat.indptr[i + 1]]:
                if j2 in pos and pos[j2] != k:
                    src.append(k)
                    dst.append(pos[j2])
        comp = _strong_components(GraphObject(True, ns, src, dst))
        for c in range(int(comp.max()) + 1):
            members = [k for k in range(ns) if comp[k] == c]
            blocks_p.extend(rows_s[k] for k in members)
            blocks_q.extend(cols_s[k] for k in members)
            bounds_r.append(bounds_r[-1] + len(members))
            bounds_s.append(bounds_s[-1] + len(members))
    p = rows_h_order + blocks_p + rows_v_order
    q = h_un + h_ma + blocks_q + v_ma
    r = [0] + bounds_r
    s = [0] + bounds_s
    # drop an empty leading (horizontal) block, add the trailing (vertical) block when it exists
    if r[1] == 0 and s[1] == 0:
        r, s = r[1:], s[1:]
    if rows_v_order or v_ma:
        r.append(m)
        s.append(n)
    cc = [0, len(h_un), len(h_un) + len(h_ma), len(h_un) + len(h_ma) + ns, n]
    rr = [0, len(rows_h_order), len(rows_h_order) + ns, len(rows_h_order) + ns + len(v_ma), m]
    outs = (_row(np.array(p) + 1.0), _row(np.array(q) + 1.0), _row(np.array(r) + 1.0), _row(np.array(s) + 1.0),
            _row(np.array(cc) + 1.0), _row(np.array(rr) + 1.0))
    return outs[:nargout]


def _etree_parent(pat, col: bool) -> np.ndarray:
    """Elimination tree of A (symmetric pattern, upper triangle used) or of A'*A (``col``)."""
    n = pat.shape[1]
    parent = -np.ones(n, dtype=np.int64)
    ancestor = -np.ones(n, dtype=np.int64)
    csc = pat.tocsc()
    prev = -np.ones(pat.shape[0], dtype=np.int64) if col else None
    for k in range(n):
        rows = csc.indices[csc.indptr[k]:csc.indptr[k + 1]]
        for i in rows:
            if col:
                i, prev[i] = prev[i], k   # the previous column with an entry in this row
            while i != -1 and i < k:
                nxt = ancestor[i]
                ancestor[i] = k
                if nxt == -1:
                    parent[i] = k
                i = nxt
    return parent


def _postorder(parent: np.ndarray) -> np.ndarray:
    n = parent.size
    children: list = [[] for _ in range(n)]
    roots = []
    for v in range(n):
        if parent[v] < 0:
            roots.append(v)
        else:
            children[parent[v]].append(v)
    order: list[int] = []
    for root in roots:
        stack = [(root, 0)]
        while stack:
            u, k = stack[-1]
            if k < len(children[u]):
                stack[-1] = (u, k + 1)
                stack.append((children[u][k], 0))
            else:
                stack.pop()
                order.append(u)
    return np.array(order, dtype=np.int64)


@builtin("etree")
def _etree(interp, args, nargout):
    if not args or len(args) > 2:
        raise MatlabError("etree expects a matrix and an optional type ('sym', 'col', 'row' or 'lo').")
    a = _pattern(args[0], "etree")
    kind = "sym"
    if len(args) > 1:
        kind = _choice(args[1], ("sym", "col", "row", "lo"), "etree", "the type")
    pat = (a != 0).tocsc()
    if kind in ("sym", "lo"):
        if pat.shape[0] != pat.shape[1]:
            raise MatlabError("etree: the matrix must be square for the 'sym' and 'lo' types.")
        import scipy.sparse as sp
        tri = sp.triu(pat) if kind == "sym" else sp.tril(pat).T
        parent = _etree_parent(sp.csc_matrix(tri), col=False)
    else:
        parent = _etree_parent(pat if kind == "col" else pat.T.tocsc(), col=True)
    out = _row(parent + 1.0)
    if nargout > 1:
        return out, _row(_postorder(parent) + 1.0)
    return out


@builtin("colperm")
def _colperm(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("colperm expects one matrix.")
    a = _pattern(args[0], "colperm")
    counts = np.asarray((a != 0).sum(axis=0)).reshape(-1)
    return _row(np.argsort(counts, kind="stable") + 1.0)


@builtin("treelayout")
def _treelayout(interp, args, nargout):
    if not args or len(args) > 2:
        raise MatlabError("treelayout expects a parent vector and an optional postorder permutation.")
    parent = np.real(fnum(args[0])).reshape(-1).astype(np.int64) - 1
    n = parent.size
    if np.any(parent >= n) or np.any(parent < -1):
        raise MatlabError("treelayout: parent entries must be 0 (a root) or node numbers.")
    post = np.real(fnum(args[1])).reshape(-1).astype(np.int64) - 1 if len(args) > 1 else _postorder(parent)
    children: list = [[] for _ in range(n)]
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)
    depth = np.zeros(n, dtype=np.int64)
    for v in post[::-1]:
        if parent[v] >= 0:
            depth[v] = depth[parent[v]] + 1
    height = int(depth.max()) + 1 if n else 0
    # leaves are spread evenly; an inner node sits above the middle of its children
    x = np.zeros(n)
    leaves = [v for v in post if not children[v]]
    for k, v in enumerate(leaves):
        x[v] = (k + 1.0) / (len(leaves) + 1.0)
    for v in post:
        if children[v]:
            x[v] = (min(x[c] for c in children[v]) + max(x[c] for c in children[v])) / 2.0
    y = (height - depth) / (height + 1.0) if n else np.zeros(0)
    outs = (_row(x), _row(y), float(height), float(max((len(c) for c in children), default=0)))
    return outs[:max(nargout, 1)] if nargout > 1 else outs[0]


@builtin("treeplot")
def _treeplot(interp, args, nargout):
    if not args or len(args) > 3:
        raise MatlabError("treeplot expects a parent vector and optional node and edge line styles.")
    x, y, _, _ = _treelayout(interp, [args[0]], 4)
    parent = np.real(fnum(args[0])).reshape(-1).astype(np.int64) - 1
    xs: list = []
    ys: list = []
    for v in range(parent.size):
        if parent[v] >= 0:
            xs.extend([x[0, v], x[0, parent[v]], np.nan])
            ys.extend([y[0, v], y[0, parent[v]], np.nan])
    node_spec = args[1] if len(args) > 1 else char("o")
    edge_spec = args[2] if len(args) > 2 else char("-")
    frame = interp.frame
    held = bool(np.all(np.asarray(interp.call_named("ishold", [], 1, frame)[0]) != 0))
    interp.call_named("plot", [_row(xs), _row(ys), edge_spec], 0, frame)
    interp.call_named("hold", [char("on")], 0, frame)
    try:
        interp.call_named("plot", [x, y, node_spec], 0, frame)
    finally:
        if not held:
            interp.call_named("hold", [char("off")], 0, frame)
    return None


@builtin("etreeplot")
def _etreeplot(interp, args, nargout):
    if not args or len(args) > 3:
        raise MatlabError("etreeplot expects a matrix and optional node and edge line styles.")
    parent = _etree(interp, [args[0]], 1)
    return _treeplot(interp, [parent, *args[1:]], nargout)


# -- public function names -----------------------------------------------------------------------------------

def _dispatcher(name: str):
    def f(interp, args, nargout):
        for a in args:
            if isinstance(a, GraphObject):
                return _METHODS[name](interp, list(args), nargout)
        from ..values import class_name
        kind = class_name(args[0]) if args else "double"
        raise MatlabError(f"Undefined function '{name}' for input arguments of type '{kind}'. "
                          "It needs a graph or digraph object.", "MATLAB:UndefinedFunction")
    f.__name__ = f"_graph_{name}"
    return f


for _name in _PUBLIC:
    if _name not in BUILTINS:
        builtin(_name)(_dispatcher(_name))


def _register_sparse_aware() -> None:
    from . import sparse as sm
    sm._OWN.update({"graph", "digraph", "dmperm", "etree", "colperm", "gplot", "etreeplot"})


_register_sparse_aware()
