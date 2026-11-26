"""Helpers shared by the statistics modules (stats_dist, stats_desc, stats_tests, stats_models,
stats_cluster, stats_ml): option parsing, labels, tables and a small base class for result objects."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, to_matlab, to_str
from ._util import fnum, is_text, text


# -- small conversions ---------------------------------------------------------------------------

def rng_of(interp) -> np.random.Generator:
    rng = interp.state.get("rng")
    if rng is None:
        rng = np.random.default_rng()
        interp.state["rng"] = rng
    return rng


def is_empty_arg(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.size == 0 and not is_char(v)


def given(args: list, i: int) -> bool:
    return len(args) > i and not is_empty_arg(args[i])


def scalar(v: Any, what: str = "argument") -> float:
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"Expected {what} to be a scalar.")
    return float(np.real(a.flat[0]))


def intval(v: Any, what: str = "argument") -> int:
    x = scalar(v, what)
    if not np.isfinite(x) or x != int(x):
        raise MatlabError(f"Expected {what} to be an integer.")
    return int(x)


def as_bool(v: Any, what: str = "option") -> bool:
    if is_text(v):
        s = text(v).lower()
        if s in ("on", "true", "yes"):
            return True
        if s in ("off", "false", "no"):
            return False
        raise MatlabError(f"{what} must be true/false or 'on'/'off'.")
    a = np.asarray(v)
    if a.size != 1:
        raise MatlabError(f"{what} must be a logical scalar.")
    return bool(a.flat[0])


def vec(v: Any) -> np.ndarray:
    """1-D float vector (column-major order)."""
    return np.asarray(fnum(v), dtype=float).flatten(order="F")


def mat(v: Any) -> np.ndarray:
    """2-D float matrix."""
    a = np.asarray(fnum(v), dtype=float)
    if a.ndim != 2:
        raise MatlabError("Expected a matrix.")
    return a


def col(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1, 1)


def row(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(1, -1)


def like_vec(values: Any, like: Any) -> np.ndarray:
    """Shape a result vector as a row when ``like`` was a row vector, else as a column."""
    v = np.asarray(values)
    shape = getattr(like, "shape", (0, 1))
    return v.reshape(1, -1) if (len(shape) == 2 and shape[0] == 1 and shape[1] != 1) else v.reshape(-1, 1)


def outs(nargout: int, *vals):
    """Return the first max(nargout, 1) values (a single value when only one is asked for)."""
    return vals[0] if nargout <= 1 else tuple(vals[:nargout])


def cellstr(items: list, column: bool = True) -> Cell:
    return Cell.from_list([char(str(s)) for s in items], column=column)


def text_list(v: Any) -> list[str] | None:
    """List of strings for a cellstr, string array, string scalar or char matrix; None otherwise."""
    if isinstance(v, MString):
        return [str(v)]
    if isinstance(v, StringArray):
        return ["" if x is None else str(x) for x in v.values_f()]
    if isinstance(v, Cell):
        out = []
        for x in v.values_f():
            if isinstance(x, MString) or is_char(x):
                out.append(to_str(x))
            elif isinstance(x, np.ndarray) and x.size == 0:
                out.append("")
            else:
                return None
        return out
    if is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return ["".join(r).rstrip() for r in v]
        return [to_str(v)]
    return None


def num_text(x: float) -> str:
    x = float(x)
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.5g}"


def struct(**fields) -> Struct:
    return Struct.scalar({k: to_matlab(v) if not isinstance(v, (Cell, Struct)) else v for k, v in fields.items()})


# -- name-value options --------------------------------------------------------------------------

def match_name(s: str, known) -> str | None:
    """Case-insensitive match of ``s`` against option names; unique prefixes are accepted."""
    low = s.lower()
    if not low:
        return None
    hits = []
    for k in known:
        kl = k.lower()
        if kl == low:
            return kl
        if kl.startswith(low):
            hits.append(kl)
    return hits[0] if len(hits) == 1 else None


def get_opts(args: list, start: int, known, fname: str) -> tuple[list, dict]:
    """Split ``args`` into positional values and name-value options.

    Scanning starts at ``start``; the options begin at the first text argument that names a
    known option. Unknown option names raise an error. Keys of the result are lower case.
    """
    pos = list(args[:start])
    i = start
    while i < len(args):
        a = args[i]
        if is_text(a) and match_name(text(a), known) is not None:
            break
        pos.append(a)
        i += 1
    rest = args[i:]
    if len(rest) % 2:
        raise MatlabError(f"{fname}: name-value arguments must come in pairs.")
    opts: dict[str, Any] = {}
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]):
            raise MatlabError(f"{fname}: expected an option name.")
        key = match_name(text(rest[k]), known)
        if key is None:
            raise MatlabError(f"{fname}: unknown option '{text(rest[k])}'.")
        opts[key] = rest[k + 1]
    return pos, opts


def choice(v: Any, allowed, what: str) -> str:
    """Validate a text option against allowed values (case-insensitive, unique prefix)."""
    if not is_text(v):
        raise MatlabError(f"{what} must be text.")
    key = match_name(text(v), allowed)
    if key is None:
        raise MatlabError(f"{what} must be one of: " + ", ".join(f"'{a}'" for a in allowed) + ".")
    return key


def no_unsupported(opts: dict, names, fname: str) -> None:
    for n in names:
        if n.lower() in opts:
            raise MatlabError(f"{fname}: the option '{n}' is not supported.")


# -- calling MATLAB functions ----------------------------------------------------------------------

def call(interp, f: Any, args: list, nargout: int = 1) -> list:
    if not isinstance(f, FunctionHandle) and not is_text(f):
        raise MatlabError("Expected a function handle or a function name.")
    vals = interp.call_function_value(f, list(args), nargout)
    if nargout >= 1 and not vals:
        raise MatlabError("The function must return a value.")
    return vals


def call_num(interp, f: Any, *args) -> np.ndarray:
    return np.asarray(fnum(call(interp, f, list(args), 1)[0]), dtype=float)


def builtin_call(interp, name: str, args: list, nargout: int = 1) -> list:
    fn = interp.builtins.get(name)
    if fn is None:
        raise MatlabError(f"Unrecognized function '{name}'.")
    return interp.call_builtin(name, fn, list(args), nargout)


# -- tables and categoricals -----------------------------------------------------------------------

def is_table(v: Any) -> bool:
    return isinstance(v, MObject) and v.class_name() == "table"


def is_categorical(v: Any) -> bool:
    return getattr(v, "kind", None) == "categorical"


def make_table(names: list[str], columns: list, row_names: list[str] | None = None):
    from .table import TableObject
    cols = []
    for c in columns:
        if isinstance(c, np.ndarray) and not is_char(c):
            c = c.reshape(-1, 1) if c.ndim == 1 else c
        elif isinstance(c, list):
            c = cellstr(c)
        cols.append(c)
    return TableObject(list(names), cols, row_names=row_names)


def make_categorical(codes: np.ndarray, cats: list[str]):
    """Categorical column from 0-based codes (-1 = undefined)."""
    from .categorical import CategoricalArray
    codes = np.asarray(codes)
    data = np.where(codes < 0, np.nan, codes + 1.0).reshape(-1, 1)
    return CategoricalArray(data, list(cats))


# -- class labels and grouping variables -------------------------------------------------------------

class Labels:
    """Class labels of any supported type, coded as integers 0..K-1 (-1 = missing)."""

    def __init__(self, v: Any, order: Any = None):
        keys, self.kind, self.dtype = _label_keys(v)
        self.n = len(keys)
        if order is not None and not is_empty_arg(order):
            okeys, okind, _ = _label_keys(order)
            if (okind == "numeric") != (self.kind in ("numeric", "logical")) and okind != self.kind \
                    and not (okind in _TEXT_KINDS and self.kind in _TEXT_KINDS):
                raise MatlabError("ClassNames must have the same type as the class labels.")
            classes = [k for k in okeys if k is not None]
        elif self.kind == "categorical":
            present = {k for k in keys if k is not None}
            classes = [c for c in v.categories if c in present]
        else:
            classes = sorted({k for k in keys if k is not None})
        self.classes = classes
        self.cats = list(v.categories) if self.kind == "categorical" else None
        lookup = {k: i for i, k in enumerate(classes)}
        self.codes = np.array([lookup.get(k, -1) if k is not None else -1 for k in keys], dtype=int)

    @property
    def k(self) -> int:
        return len(self.classes)

    @property
    def names(self) -> list[str]:
        if self.kind in ("numeric", "logical"):
            return [num_text(c) for c in self.classes]
        return [str(c) for c in self.classes]

    def encode(self, v: Any) -> np.ndarray:
        keys, kind, _ = _label_keys(v)
        if (kind in ("numeric", "logical")) != (self.kind in ("numeric", "logical")):
            raise MatlabError("The class labels must have the same type as the labels used for training.")
        lookup = {k: i for i, k in enumerate(self.classes)}
        return np.array([lookup.get(k, -1) if k is not None else -1 for k in keys], dtype=int)

    def decode(self, codes: Any) -> Any:
        """Labels (same type as the originals) for an array of class codes, as a column."""
        codes = np.asarray(codes, dtype=int).flatten()
        if self.kind == "numeric":
            vals = np.array([self.classes[c] if c >= 0 else np.nan for c in codes], dtype=float).reshape(-1, 1)
            if self.dtype is not None and self.dtype.kind in "iu":
                return vals.astype(self.dtype)
            return vals
        if self.kind == "logical":
            return np.array([bool(self.classes[c]) if c >= 0 else False for c in codes]).reshape(-1, 1)
        names = [str(self.classes[c]) if c >= 0 else "" for c in codes]
        if self.kind == "categorical":
            cats = self.cats if self.cats is not None else [str(c) for c in self.classes]
            idx = np.array([cats.index(n) if n in cats else -1 for n in names])
            return make_categorical(idx, cats)
        if self.kind == "string":
            d = np.empty((len(names), 1), dtype=object)
            for i, s in enumerate(names):
                d[i, 0] = s
            from ..values import string_result
            return string_result(d)
        if self.kind == "char":
            width = max((len(s) for s in names), default=0)
            if not names:
                return np.zeros((0, 0), dtype="<U1")
            return np.array([list(s.ljust(width)) for s in names], dtype="<U1").reshape(len(names), width)
        return cellstr(names)

    def class_value(self) -> Any:
        """The ClassNames property."""
        return self.decode(np.arange(self.k))


_TEXT_KINDS = ("cellstr", "char", "string")


def _label_keys(v: Any) -> tuple[list, str, Any]:
    if is_categorical(v):
        return list(v.names_f()), "categorical", None
    if isinstance(v, (MString, StringArray)):
        items = [str(v)] if isinstance(v, MString) else list(v.values_f())
        return [None if (s is None or s == "") else str(s) for s in items], "string", None
    if isinstance(v, Cell):
        items = text_list(v)
        if items is None:
            raise MatlabError("Class labels in a cell array must all be text.")
        return [None if s == "" else s for s in items], "cellstr", None
    if is_char(v):
        if v.ndim == 2 and v.shape[0] >= 1 and v.shape[1] >= 1:
            items = ["".join(r).rstrip() for r in v]
        else:
            items = []
        return [None if s == "" else s for s in items], "char", None
    if isinstance(v, np.ndarray):
        flat = v.flatten(order="F")
        if v.dtype == bool:
            return [float(x) for x in flat], "logical", v.dtype
        fl = flat.astype(float)
        return [None if np.isnan(x) else float(x) for x in fl], "numeric", v.dtype
    raise MatlabError("Labels must be numeric, logical, categorical, text or a cell array of text.")


def group_codes(g: Any) -> tuple[np.ndarray, list[str], Any]:
    """0-based group codes (-1 = missing), group names and group levels of a grouping variable."""
    from .ndmath import group_index
    idx, names, levels = group_index(g)
    codes = np.where(np.isnan(idx), -1, idx - 1).astype(int)
    return codes, names, levels


def multi_group(gs: list, sep: str = "_") -> tuple[np.ndarray, list[str], list[list[str]], list[np.ndarray]]:
    """Combine several grouping variables.

    Returns the combined 0-based codes (-1 = missing in any variable), the combined names, the list
    of names per variable for each combined group and the per-variable codes of each combined group.
    Combined groups are ordered with the first variable varying slowest; unused combinations are dropped.
    """
    parts = [group_codes(g) for g in gs]
    n = parts[0][0].size
    for c, _, _ in parts:
        if c.size != n:
            raise MatlabError("Grouping variables must all have the same number of elements.")
    sizes = [max(len(p[1]), 1) for p in parts]
    combo = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    for (c, _, _), s in zip(parts, sizes):
        bad |= c < 0
        combo = combo * s + np.where(c < 0, 0, c)
    combo[bad] = -1
    used = np.unique(combo[~bad])
    remap = {int(u): i for i, u in enumerate(used)}
    codes = np.array([remap[int(c)] if c >= 0 else -1 for c in combo], dtype=int)
    names, per_var, per_codes = [], [], []
    for u in used:
        digits = []
        r = int(u)
        for s in reversed(sizes):
            digits.append(r % s)
            r //= s
        digits = digits[::-1]
        labs = [parts[j][1][d] for j, d in enumerate(digits)]
        per_var.append(labs)
        per_codes.append(np.array(digits))
        names.append(sep.join(labs))
    return codes, names, per_var, per_codes


def group_list(g: Any) -> list:
    """A grouping argument as a list of grouping variables (a cell of variables or a single one)."""
    if isinstance(g, Cell) and text_list(g) is None:
        return list(g.values_f())
    return [g]


# -- result objects ----------------------------------------------------------------------------------

class StatObject(MObject):
    """Base for the value objects of the statistics functions.

    Stored properties live in ``self.p`` (insertion order is the display order); a method named
    ``g_<Prop>`` computes a property on demand and a method named ``m_<name>(interp, args, nargout)``
    is a MATLAB method (callable as ``name(obj, ...)`` or ``obj.name(...)``).
    """

    matlab_class = "StatObject"
    also_isa: tuple = ()
    header: str | None = None

    def __init__(self):
        self.p: dict[str, Any] = {}

    def __init_subclass__(cls, **kw):
        # Property getters of subclasses may return plain Python numbers and strings.
        super().__init_subclass__(**kw)
        gf = cls.__dict__.get("get_field")
        if gf is not None:
            def wrapped(self, interp, name, _gf=gf):
                return to_matlab(_gf(self, interp, name))
            cls.get_field = wrapped

    def class_name(self) -> str:
        return self.matlab_class

    def isa_class(self, name: str) -> bool:
        return name == self.matlab_class or name in self.also_isa

    def has_method(self, name: str) -> bool:
        return callable(getattr(self, "m_" + name, None))

    def method_names(self) -> list[str]:
        return sorted(n[2:] for n in dir(self) if n.startswith("m_") and callable(getattr(self, n)))

    def property_names(self) -> list[str]:
        computed = [n[2:] for n in dir(self) if n.startswith("g_") and callable(getattr(self, n))]
        return list(self.p) + [c for c in computed if c not in self.p]

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        fn = getattr(self, "m_" + name, None)
        if fn is None:
            raise MatlabError(f"No method '{name}' for class '{self.class_name()}'.")
        rest = list(args[1:]) if args and args[0] is self else [a for a in args if a is not self]
        res = fn(interp, rest, nargout)
        if res is None:
            return []
        if isinstance(res, tuple):
            return [to_matlab(r) for r in res]
        return [to_matlab(res)]

    def get_field(self, interp, name: str) -> Any:
        if name in self.p:
            return to_matlab(self.p[name])
        g = getattr(self, "g_" + name, None)
        if g is not None:
            return to_matlab(g(interp))
        if self.has_method(name):
            vals = self.call_method(interp, name, [self], 1)
            if not vals:
                raise MatlabError(f"'{name}' does not return a value.")
            return vals[0]
        raise MatlabError(f"Unrecognized property or method '{name}' for class '{self.class_name()}'.")

    def set_field(self, interp, name: str, value: Any):
        raise MatlabError(f"Cannot set property '{name}' of class '{self.class_name()}'.")

    def clone(self):
        import copy
        other = copy.copy(self)
        other.p = dict(self.p)
        return other

    def display_lines(self) -> list[str]:
        from ..display import summary
        short = self.matlab_class.rsplit(".", 1)[-1]
        lines = [f"  {self.header or short}", ""] if self.header is None else [f"  {short}", "", f"  {self.header}"]
        names = [n for n in self.p if not n.startswith("_")]
        if names:
            if self.header is None:
                lines = [f"  {short} with properties:", ""]
            width = max(len(n) for n in names)
            for n in names:
                v = to_matlab(self.p[n])
                lines.append(f"    {n.rjust(width)}: {_brief(v, summary)}")
        return lines


def _brief(v: Any, summary: Callable) -> str:
    if isinstance(v, np.ndarray) and not is_char(v) and v.ndim == 2 and v.shape[0] == 1 and 0 < v.size <= 8:
        if v.dtype == bool:
            return " ".join("1" if x else "0" for x in v.flat) if v.size > 1 else ("1" if v.flat[0] else "0")
        body = " ".join(_gnum(x) for x in v.flat)
        return body if v.size == 1 else f"[{body}]"
    if is_char(v) and v.ndim == 2 and v.shape[0] <= 1:
        return f"'{to_str(v)}'"
    try:
        return summary(v, True)
    except Exception:  # pragma: no cover - defensive: summary of an unusual value
        return f"[{type(v).__name__}]"


def _gnum(x: Any) -> str:
    x = float(np.real(x))
    if np.isnan(x):
        return "NaN"
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == int(x) and abs(x) < 1e10:
        return str(int(x))
    return f"{x:.5g}"


# -- numeric helpers -----------------------------------------------------------------------------------

def num_hessian(f: Callable, x: np.ndarray, rel: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = rel * np.maximum(np.abs(x), 1e-2)
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def num_jacobian(f: Callable, x: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector function of a vector."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float).ravel()
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        h = rel * max(abs(x[i]), 1e-3)
        e = np.zeros(x.size)
        e[i] = h
        J[:, i] = (np.asarray(f(x + e), dtype=float).ravel() - np.asarray(f(x - e), dtype=float).ravel()) / (2 * h)
    return J


def tied_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks of a 1-D array (NaN stays NaN and is ranked last / ignored)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = ~np.isnan(x)
    v = x[ok]
    if v.size:
        order = np.argsort(v, kind="mergesort")
        s = v[order]
        ranks = np.empty(v.size)
        i = 0
        while i < s.size:
            j = i
            while j + 1 < s.size and s[j + 1] == s[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        out[ok] = ranks
    return out


def table_columns(tbl) -> dict:
    return dict(zip(tbl.names, tbl.columns))
