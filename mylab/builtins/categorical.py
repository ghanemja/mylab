"""categorical arrays: values drawn from a fixed list of named categories.

A categorical array stores, for each element, the 1-based number of its
category (NaN for <undefined>) in ``data``, next to the list of category names.
That is the same layout datetime arrays use, so tables, grouping functions and
sorting treat it as a column of numbers ordered like the categories.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .. import indexing as ix
from .. import ops
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, StringArray, as_2d, char, is_char, to_matlab, to_str
from ._registry import builtin
from ._util import bool_scalar, is_text, nargs, text, to_int

UNDEFINED = "<undefined>"
_NATIVE: dict[str, Callable] = {}


def native(*names: str):
    def deco(fn: Callable) -> Callable:
        for n in names:
            _NATIVE[n] = fn
        return fn
    return deco


def _size_text(shape: tuple) -> str:
    return "×".join(str(d) for d in shape)


def _number_name(x: float) -> str:
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else f"{x:.5g}"


def _texts(v: Any) -> tuple[list, tuple] | None:
    """Text items (None for missing) and shape of a text-like value; None if it is not text."""
    if isinstance(v, MString):
        return [str(v)], (1, 1)
    if isinstance(v, StringArray):
        return list(v.data.flatten(order="F")), v.data.shape
    if is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return ["".join(r).strip() for r in v], (v.shape[0], 1)
        return [to_str(v).strip()], (1, 1)
    if isinstance(v, Cell):
        items = []
        for x in v.data.flatten(order="F"):
            if isinstance(x, MString) or is_char(x):
                items.append(to_str(x).strip())
            elif isinstance(x, np.ndarray) and x.size == 0:
                items.append("")
            else:
                return None
        return items, v.data.shape
    return None


class CategoricalArray(MObject):
    kind = "categorical"

    def __init__(self, data: Any, categories: list[str], ordinal: bool = False):
        a = np.asarray(data, dtype=float)
        self.data = as_2d(a) if a.ndim != 2 else a
        self.categories = list(categories)
        self.ordinal = ordinal

    # -- array duck type (shared with datetime arrays; used by tables and for-loops) -----------------
    def matlab_shape(self) -> tuple:
        return tuple(self.data.shape)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def codes(self) -> np.ndarray:
        """0-based category numbers, -1 for undefined."""
        return np.where(np.isnan(self.data), -1, self.data - 1).astype(int)

    def like(self, data: Any) -> "CategoricalArray":
        return CategoricalArray(data, self.categories, self.ordinal)

    def copy(self) -> "CategoricalArray":
        return self.like(self.data.copy())

    def empty_like(self) -> "CategoricalArray":
        return self.like(np.zeros((0, 0)))

    def reshape(self, shape, order="F") -> "CategoricalArray":
        return self.like(self.data.reshape(shape, order=order))

    def __getitem__(self, key) -> "CategoricalArray":
        return self.like(self.data[key])

    def class_name(self) -> str:
        return "categorical"

    def isa_class(self, name: str) -> bool:
        return name == "categorical"

    def numeric_error(self, what: str) -> str:
        return ("A categorical array cannot be used as a number here. Use double(c) for the category "
                "numbers, or compare with a category name, e.g. c == 'red'.")

    def has_method(self, name: str) -> bool:
        return name in _NATIVE

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        res = _NATIVE[name](interp, args, nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    # -- names and codes --------------------------------------------------------------------------
    def element_texts(self) -> np.ndarray:
        out = np.empty(self.data.shape, dtype=object)
        for idx in np.ndindex(self.data.shape):
            c = self.data[idx]
            out[idx] = UNDEFINED if np.isnan(c) else self.categories[int(c) - 1]
        return out

    def names_f(self) -> list:
        """Category name of each element in column-major order (None for undefined)."""
        return [None if np.isnan(c) else self.categories[int(c) - 1] for c in self.data.flatten(order="F")]

    def codes_for(self, value: Any, extend: bool) -> tuple[np.ndarray, list[str]]:
        """Codes of ``value`` in this array's categories; with ``extend`` new names become categories."""
        cats = list(self.categories)
        if isinstance(value, CategoricalArray):
            names, shape = value.names_f(), value.data.shape
        else:
            found = _texts(value)
            if found is None:
                if isinstance(value, np.ndarray) and value.size == 0:
                    return np.zeros((0, 0)), cats
                raise MatlabError("Only text, a cell array of text or another categorical array can be "
                                  "combined with a categorical array.")
            names, shape = found
        codes = np.empty(len(names))
        for k, n in enumerate(names):
            if n is None or n == "" or n == UNDEFINED:
                codes[k] = np.nan
            elif n in cats:
                codes[k] = cats.index(n) + 1
            elif extend and not self.ordinal:
                cats.append(n)
                codes[k] = len(cats)
            elif self.ordinal:
                raise MatlabError(f"'{n}' is not a category of this ordinal categorical array. Its "
                                  "categories are: " + ", ".join(self.categories) + ".")
            else:
                codes[k] = np.nan if not extend else codes[k]
                if not extend:
                    raise MatlabError(f"'{n}' is not one of the categories: " + ", ".join(cats) + ".")
        return codes.reshape(shape, order="F"), cats

    def coerce(self, value: Any) -> np.ndarray:
        return self.codes_for(value, extend=False)[0]

    # -- indexing and concatenation ------------------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        return self.like(ix.get(self.data, list(args)))

    def index_set(self, interp, args: list, value: Any) -> Any:
        codes, cats = self.codes_for(value, extend=True)
        return CategoricalArray(ix.set_(self.data.copy(), list(args), codes, lambda: np.nan), cats, self.ordinal)

    def index_delete(self, interp, args: list) -> Any:
        return self.like(ix.delete(self.data, list(args)))

    def brace_get(self, interp, args: list) -> list:
        raise MatlabError("Brace indexing is not supported for categorical arrays. Use parentheses: c(k).")

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        first = next(i for i in items if isinstance(i, CategoricalArray))
        result = first.like(np.zeros((0, 0)))
        datas = []
        for i in items:
            if isinstance(i, np.ndarray) and i.size == 0 and not is_char(i):
                continue
            if isinstance(i, CategoricalArray) and i.ordinal != first.ordinal:
                raise MatlabError("Cannot concatenate an ordinal categorical array with one that is not.")
            if isinstance(i, CategoricalArray) and first.ordinal and i.categories != first.categories:
                raise MatlabError("Ordinal categorical arrays must have the same categories, in the same "
                                  "order, to be concatenated.")
            codes, cats = result.codes_for(i, extend=True)
            result = CategoricalArray(result.data, cats, first.ordinal)
            datas.append(codes)
        return result.like(ops._concat_arrays(datas, axis)) if datas else result

    def object_equals(self, other: Any) -> bool:
        if not isinstance(other, CategoricalArray) or other.data.shape != self.data.shape:
            return False
        a, b = self.names_f(), other.names_f()
        return all(x == y and x is not None for x, y in zip(a, b))

    # -- operators ----------------------------------------------------------------------------------
    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("==", "~=", "<", "<=", ">", ">="):
            return NotImplemented
        cat = left if isinstance(left, CategoricalArray) else right
        ordering = op not in ("==", "~=")
        if ordering and not cat.ordinal:
            raise MatlabError(f"Relational comparison ({op}) needs an ordinal categorical array. Create one "
                              "with categorical(values, categoryOrder, 'Ordinal', true).")

        def codes_of(v: Any) -> np.ndarray:
            if v is cat:
                return cat.data
            if isinstance(v, CategoricalArray):
                if ordering and v.categories != cat.categories:
                    raise MatlabError("Ordinal categorical arrays must have the same categories to be "
                                      "compared.")
                names, shape = v.names_f(), v.data.shape
            else:
                found = _texts(v)
                if found is None:
                    raise MatlabError("A categorical array can be compared with text, a cell array of "
                                      "text or another categorical array.")
                names, shape = found
            out = np.array([cat.categories.index(n) + 1 if n in cat.categories else
                            (np.nan if (n is None or n in ("", UNDEFINED)) else -1.0) for n in names])
            if ordering and np.any(out == -1):
                bad = next(n for n in names if n not in cat.categories and n not in (None, "", UNDEFINED))
                raise MatlabError(f"'{bad}' is not a category of this ordinal categorical array.")
            return out.reshape(shape, order="F")

        a, b = codes_of(left), codes_of(right)
        try:
            a, b = np.broadcast_arrays(a, b)
        except ValueError:
            raise MatlabError("Arrays have incompatible sizes for this operation.") from None
        with np.errstate(invalid="ignore"):
            fn = {"==": np.equal, "~=": np.not_equal, "<": np.less, "<=": np.less_equal,
                  ">": np.greater, ">=": np.greater_equal}[op]
            return fn(a, b)

    def unary_op(self, interp, op: str) -> Any:
        if op in ("'", ".'"):
            return self.like(self.data.T.copy())
        return NotImplemented

    # -- display --------------------------------------------------------------------------------------
    def summary_text(self, in_cell: bool) -> str:
        if self.data.size == 1:
            return str(self.element_texts().flat[0])
        s = f"{_size_text(self.data.shape)} categorical"
        return s if in_cell else f"[{s}]"

    def display_lines(self) -> list[str]:
        if self.data.size == 0:
            return []
        if self.data.ndim > 2:
            return [f"  {_size_text(self.data.shape)} categorical array"]
        texts = self.element_texts()
        width = max(len(s) for s in texts.flat)
        return ["     " + "      ".join(str(s).ljust(width) for s in row).rstrip() + " " for row in texts]

    def format_display(self, name: str) -> str:
        shape = self.data.shape
        kind = "ordinal categorical" if self.ordinal and self.data.size != 1 else "categorical"
        if self.data.size == 0:
            return f"{name} = \n\n  {_size_text(shape)} empty categorical array\n\n"
        head = f"  {kind}" if self.data.size == 1 else f"  {_size_text(shape)} {kind} array"
        return f"{name} = \n\n{head}\n\n" + "\n".join(self.display_lines()) + "\n\n"


# -- construction -------------------------------------------------------------------------------------

def _value_keys(v: Any, what: str) -> tuple[list, tuple, bool]:
    """Comparable keys (None = missing) for the elements of v, its shape, and whether v is numeric."""
    if isinstance(v, CategoricalArray):
        return v.names_f(), v.data.shape, False
    found = _texts(v)
    if found is not None:
        items, shape = found
        return [None if (x is None or x == "" or x == UNDEFINED) else x for x in items], shape, False
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        flat = v.flatten(order="F")
        if v.dtype == bool:
            return [bool(x) for x in flat], v.shape, True
        return [None if np.isnan(x) else float(x) for x in flat.astype(float)], v.shape, True
    raise MatlabError(f"{what}: the values must be numbers, logical values, text, a cell array of text or "
                      "a categorical array.")


def _key_name(k: Any) -> str:
    if isinstance(k, bool):
        return "true" if k else "false"
    if isinstance(k, float):
        return _number_name(k)
    return str(k)


def make_categorical(args: list) -> CategoricalArray:
    rest = list(args)
    ordinal = False
    k = len(rest) - 2
    while k >= 1 and is_text(rest[k]) and text(rest[k]).lower() in ("ordinal", "protected"):
        if text(rest[k]).lower() == "ordinal":
            v = rest[k + 1]
            ordinal = text(v).lower() in ("true", "on") if is_text(v) else bool(np.asarray(v).flat[0])
        del rest[k:k + 2]
        k = len(rest) - 2
    if not rest:
        return CategoricalArray(np.zeros((0, 0)), [])
    if len(rest) > 3:
        raise MatlabError("categorical takes the values, an optional list of allowed values and optional "
                          "category names.")
    source = rest[0]
    keys, shape, numeric = _value_keys(source, "categorical")
    if isinstance(source, CategoricalArray) and len(rest) == 1:
        return CategoricalArray(source.data.copy(), source.categories, ordinal or source.ordinal)
    if len(rest) >= 2:
        value_keys, _, _ = _value_keys(rest[1], "categorical")
        if any(v is None for v in value_keys):
            raise MatlabError("categorical: the list of allowed values must not contain missing values.")
        if len(set(value_keys)) != len(value_keys):
            raise MatlabError("categorical: the list of allowed values contains duplicates.")
    elif isinstance(source, CategoricalArray):
        value_keys = list(source.categories)
    else:
        present = {x for x in keys if x is not None}
        value_keys = sorted(present)
    if len(rest) == 3:
        found = _texts(rest[2])
        if found is None or len(found[0]) != len(value_keys):
            raise MatlabError("categorical: give one category name for each allowed value.")
        names = [str(n) for n in found[0]]
    else:
        names = [_key_name(v) for v in value_keys]
    # Several values may share one name (categorical(x, [1 2 3], {'low', 'low', 'high'})).
    cats: list[str] = []
    for n in names:
        if n not in cats:
            cats.append(n)
    lookup = {v: cats.index(n) + 1.0 for v, n in zip(value_keys, names)}
    data = np.array([lookup.get(x, np.nan) if x is not None else np.nan for x in keys], dtype=float)
    return CategoricalArray(data.reshape(shape, order="F"), cats, ordinal)


@builtin("categorical")
def _categorical(interp, args, nargout):
    return make_categorical(args)


@builtin("iscategorical")
def _iscategorical(interp, args, nargout):
    nargs(args, 1, 1, "iscategorical")
    return bool_scalar(isinstance(args[0], CategoricalArray))


def _need(name: str, example: str):
    def fn(interp, args, nargout):
        raise MatlabError(f"{name} needs a categorical array as its first argument. Example: {example}")
    builtin(name)(fn)


for _name, _example in (("categories", "categories(categorical({'a', 'b', 'a'}))"),
                        ("countcats", "countcats(categorical({'a', 'b', 'a'}))"),
                        ("isundefined", "isundefined(categorical({'a', ''}))"),
                        ("iscategory", "iscategory(c, 'a')"), ("isordinal", "isordinal(c)"),
                        ("addcats", "c = addcats(c, 'new')"), ("removecats", "c = removecats(c)"),
                        ("renamecats", "c = renamecats(c, 'old', 'new')"),
                        ("reordercats", "c = reordercats(c, {'low', 'mid', 'high'})"),
                        ("mergecats", "c = mergecats(c, {'a', 'b'}, 'ab')"),
                        ("setcats", "c = setcats(c, {'a', 'b'})")):
    _need(_name, _example)


# -- methods ----------------------------------------------------------------------------------------

def _cat(args: list) -> CategoricalArray:
    return next(a for a in args if isinstance(a, CategoricalArray))


def _name_list(v: Any, what: str) -> list[str]:
    found = _texts(v)
    if found is None:
        raise MatlabError(f"{what}: category names must be text or a cell array of text.")
    return [str(x) for x in found[0]]


def _cellstr_col(names: list[str]) -> Cell:
    return Cell.from_list([char(n) for n in names], column=True)


@native("categories")
def _categories(interp, args, nargout):
    return _cellstr_col(args[0].categories)


@native("countcats")
def _countcats(interp, args, nargout):
    c = args[0]
    n = len(c.categories)
    data = c.data
    if len(args) > 1:
        axis = to_int(args[1], "dimension") - 1
    else:
        axis = next((d for d, s in enumerate(data.shape) if s != 1), 0)
    if axis >= data.ndim:
        data = data.reshape(data.shape + (1,) * (axis + 1 - data.ndim))
    moved = np.moveaxis(data, axis, 0)
    out = np.stack([np.sum(moved == k + 1, axis=0) for k in range(n)], axis=0) if n else \
        np.zeros((0,) + moved.shape[1:])
    return as_2d(np.moveaxis(out, 0, axis).astype(float))


@native("isundefined", "ismissing", "isnan")
def _isundefined(interp, args, nargout):
    return np.isnan(args[0].data)


@native("iscategory")
def _iscategory(interp, args, nargout):
    found = _texts(args[1])
    if found is None:
        raise MatlabError("iscategory: give category names as text or a cell array of text.")
    return np.array([n in args[0].categories for n in found[0]], dtype=bool).reshape(found[1], order="F")


@native("isordinal")
def _isordinal(interp, args, nargout):
    return bool_scalar(args[0].ordinal)


@native("addcats")
def _addcats(interp, args, nargout):
    c = args[0]
    new = _name_list(args[1], "addcats")
    dup = [n for n in new if n in c.categories]
    if dup:
        raise MatlabError(f"addcats: '{dup[0]}' is already a category.")
    cats = list(c.categories)
    pos = len(cats)
    if len(args) >= 4 and is_text(args[2]):
        where, ref = text(args[2]).lower(), text(args[3])
        if ref not in cats or where not in ("before", "after"):
            raise MatlabError("addcats: use 'Before' or 'After' followed by an existing category name.")
        pos = cats.index(ref) + (1 if where == "after" else 0)
    elif c.ordinal:
        pos = len(cats)
    new_cats = cats[:pos] + new + cats[pos:]
    remap = np.array([np.nan] + [new_cats.index(n) + 1.0 for n in cats])
    return CategoricalArray(remap[np.where(np.isnan(c.data), 0, c.data).astype(int)], new_cats, c.ordinal)


def _recode(c: CategoricalArray, new_cats: list[str], mapping: dict[str, str | None]) -> CategoricalArray:
    """Rebuild with new categories; mapping sends each old name to a new name (None = undefined)."""
    remap = [np.nan]
    for old in c.categories:
        target = mapping.get(old, old)
        remap.append(new_cats.index(target) + 1.0 if target in new_cats else np.nan)
    table = np.array(remap)
    return CategoricalArray(table[np.where(np.isnan(c.data), 0, c.data).astype(int)], new_cats, c.ordinal)


@native("removecats")
def _removecats(interp, args, nargout):
    c = args[0]
    if len(args) > 1:
        gone = _name_list(args[1], "removecats")
    else:
        used = set(c.names_f())
        gone = [n for n in c.categories if n not in used]
    return _recode(c, [n for n in c.categories if n not in gone], {})


@native("renamecats")
def _renamecats(interp, args, nargout):
    c = args[0]
    if len(args) == 2:
        old, new = list(c.categories), _name_list(args[1], "renamecats")
    else:
        old, new = _name_list(args[1], "renamecats"), _name_list(args[2], "renamecats")
    if len(old) != len(new):
        raise MatlabError("renamecats: give one new name for each category that is renamed.")
    missing = [n for n in old if n not in c.categories]
    if missing:
        raise MatlabError(f"renamecats: '{missing[0]}' is not a category.")
    mapping = dict(zip(old, new))
    cats = [mapping.get(n, n) for n in c.categories]
    if len(set(cats)) != len(cats):
        raise MatlabError("renamecats: the new names must differ from each other and from the other "
                          "categories. Use mergecats to combine categories.")
    return CategoricalArray(c.data.copy(), cats, c.ordinal)


@native("reordercats")
def _reordercats(interp, args, nargout):
    c = args[0]
    if len(args) == 1:
        order = sorted(c.categories)
    elif isinstance(args[1], np.ndarray) and not is_char(args[1]):
        idx = [int(i) - 1 for i in args[1].flatten()]
        if sorted(idx) != list(range(len(c.categories))):
            raise MatlabError("reordercats: the order must be a permutation of 1 to the number of categories.")
        order = [c.categories[i] for i in idx]
    else:
        order = _name_list(args[1], "reordercats")
    if sorted(order) != sorted(c.categories):
        raise MatlabError("reordercats: the new order must list every existing category exactly once.")
    return _recode(c, order, {})


@native("mergecats")
def _mergecats(interp, args, nargout):
    c = args[0]
    old = _name_list(args[1], "mergecats")
    missing = [n for n in old if n not in c.categories]
    if missing:
        raise MatlabError(f"mergecats: '{missing[0]}' is not a category.")
    new = text(args[2]) if len(args) > 2 else old[0]
    cats: list[str] = []
    for n in c.categories:
        name = new if n in old else n
        if name not in cats:
            cats.append(name)
    return _recode(c, cats, {n: new for n in old})


@native("setcats")
def _setcats(interp, args, nargout):
    return _recode(args[0], _name_list(args[1], "setcats"), {})


def _numeric_cast(name: str):
    def fn(interp, args, nargout):
        data = args[0].data
        if name in ("double", "single"):
            return data.astype(float if name == "double" else np.float32)
        return interp.call_builtin(name, interp.builtins[name], [np.where(np.isnan(data), 0, data)], 1)[0]
    return fn


for _n in ("double", "single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
    _NATIVE[_n] = _numeric_cast(_n)


@native("cellstr")
def _cellstr(interp, args, nargout):
    c = args[0]
    texts = c.element_texts()
    data = np.empty(texts.shape, dtype=object)
    for idx in np.ndindex(texts.shape):
        data[idx] = char(texts[idx])
    return Cell(data)


@native("string")
def _string(interp, args, nargout):
    c = args[0]
    data = np.empty(c.data.shape, dtype=object)
    names = c.names_f()
    flat = data.reshape(-1, order="F")
    for k, n in enumerate(names):
        flat[k] = n
    out = flat.reshape(c.data.shape, order="F")
    return MString(out.flat[0]) if out.size == 1 and out.flat[0] is not None else StringArray(out)


@native("char")
def _char(interp, args, nargout):
    texts = [str(s) for s in args[0].element_texts().flatten(order="F")]
    if not texts:
        return char("")
    width = max(len(s) for s in texts)
    return np.array([list(s.ljust(width)) for s in texts], dtype="<U1").reshape(len(texts), width)


@native("ismember")
def _ismember(interp, args, nargout):
    a, pool = args[0], args[1]
    if not isinstance(a, CategoricalArray):
        found = _texts(a)
        if found is None:
            raise MatlabError("ismember: compare a categorical array with text or categorical values.")
        names, shape = found
    else:
        names, shape = a.names_f(), a.data.shape
    if isinstance(pool, CategoricalArray):
        options = [n for n in pool.names_f() if n is not None]
    else:
        found = _texts(pool)
        if found is None:
            raise MatlabError("ismember: the set must be text, a cell array of text or a categorical array.")
        options = [n for n in found[0] if n]
    flags = np.array([n in options for n in names], dtype=bool).reshape(shape, order="F")
    loc = np.array([options.index(n) + 1.0 if n in options else 0.0 for n in names]).reshape(shape, order="F")
    return flags, loc


@native("unique")
def _unique(interp, args, nargout):
    c = args[0]
    stable = any(is_text(a) and text(a).lower() == "stable" for a in args[1:])
    flat = c.data.flatten(order="F")
    key = np.where(np.isnan(flat), np.inf, flat)
    values, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    if stable:
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        values, first, inverse = values[order], first[order], rank[inverse]
    n_undefined = int(np.sum(np.isnan(flat)))
    if n_undefined > 1:  # each undefined element is distinct, like NaN
        keep = values != np.inf
        extra = np.flatnonzero(np.isnan(flat))
        values = np.concatenate([values[keep], np.full(extra.size, np.inf)])
        first = np.concatenate([first[keep], extra])
        inverse = inverse.astype(float)
        inverse[np.isnan(flat)] = np.arange(int(np.sum(keep)), int(np.sum(keep)) + extra.size)
    out = np.where(values == np.inf, np.nan, values)
    row = c.data.shape[0] == 1 and c.data.ndim == 2 and c.data.size > 0
    shape = (1, -1) if row else (-1, 1)
    return c.like(out.reshape(shape)), (first + 1.0).reshape(-1, 1), (inverse + 1.0).reshape(-1, 1)


@native("sort")
def _sort(interp, args, nargout):
    c = args[0]
    descend = any(is_text(a) and text(a).lower() == "descend" for a in args[1:])
    dims = [a for a in args[1:] if not is_text(a)]
    axis = to_int(dims[0], "dimension") - 1 if dims else next(
        (d for d, s in enumerate(c.data.shape) if s != 1), 0)
    if axis >= c.data.ndim:
        return c.copy(), np.ones(c.data.shape)
    key = c.data if not descend else -c.data
    order = np.argsort(np.where(np.isnan(key), np.inf, key), axis=axis, kind="stable")
    return c.like(np.take_along_axis(c.data, order, axis=axis)), order + 1.0


@native("isequal")
def _isequal(interp, args, nargout):
    first = args[0]
    return bool_scalar(all(isinstance(first, CategoricalArray) and first.object_equals(b) for b in args[1:]))


@native("histcounts")
def _histcounts(interp, args, nargout):
    c = args[0]
    counts = np.array([float(np.sum(c.data == k + 1)) for k in range(len(c.categories))]).reshape(1, -1)
    return counts, Cell.from_list([char(n) for n in c.categories])


@native("summary")
def _summary(interp, args, nargout):
    c = args[0]
    counts = [int(np.sum(c.data == k + 1)) for k in range(len(c.categories))]
    names = list(c.categories)
    n_undefined = int(np.sum(np.isnan(c.data)))
    if n_undefined:
        names.append(UNDEFINED)
        counts.append(n_undefined)
    if not names:
        interp.out("\n")
        return None
    width = max(max(len(n) for n in names), max(len(str(k)) for k in counts))
    interp.out("     " + "      ".join(n.ljust(width) for n in names).rstrip() + " \n")
    interp.out("     " + "      ".join(str(k).ljust(width) for k in counts).rstrip() + " \n\n")
    return None


@native("disp")
def _disp(interp, args, nargout):
    lines = args[0].display_lines()
    interp.out("\n".join(lines) + "\n" if lines else "")
    return None


@native("histogram", "bar")
def _histogram(interp, args, nargout):
    c = _cat(args)
    k = args.index(c)
    names = [char(n) for n in c.categories]
    if k == 0 and len(args) > 1 and isinstance(args[1], np.ndarray) and not is_char(args[1]):
        heights = args[1]  # bar(c, y)
        rest = list(args[2:])
        labels = [char(str(s)) for s in c.element_texts().flatten(order="F")]
    else:
        heights = np.array([float(np.sum(c.data == j + 1)) for j in range(len(c.categories))]).reshape(1, -1)
        rest = [a for a in args[k + 1:]]
        labels = names
    res = interp.call_builtin("bar", interp.builtins["bar"], list(args[:k]) + [heights] + rest, nargout)
    n = len(labels)
    interp.call_builtin("xticks", interp.builtins["xticks"], [np.arange(1.0, n + 1).reshape(1, -1)], 0)
    interp.call_builtin("xticklabels", interp.builtins["xticklabels"], [Cell.from_list(labels)], 0)
    return tuple(res) if res else None
