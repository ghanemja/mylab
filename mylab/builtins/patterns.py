"""Pattern objects (digitsPattern, lettersPattern, wildcardPattern ...) and the text
functions that search with them.

A pattern is kept as a small tree and compiled to a regular expression when a function
searches with it. The functions in this module accept a char vector, a cell array of char
vectors or a string array as text, and plain text, a list of texts or a pattern object as
what to look for, and they return the class MATLAB returns: logical or numeric arrays of the
size of the text, or text of the same kind as the input.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, StringArray, char, class_name, is_char, string_result, to_numeric, to_str
from ._registry import BUILTINS, builtin
from ._util import nargs, parse_options

_LETTER = r"[^\W\d_]"
_ALNUM = r"[^\W_]"
_DIGIT = r"[0-9]"
_SPACE = r"\s"


# -- the pattern tree ------------------------------------------------------------------------------
# ("lit", text) | ("set", regex for one character) | ("rep", node, lo, hi, greedy)
# ("seq", [nodes]) | ("alt", [nodes]) | ("case", node, insensitive) | ("raw", regex)
# ("bound", regex, negated regex, negated) | ("look", node, ahead, negated) | ("named", node, name)

def _rx(node: tuple) -> str:
    kind = node[0]
    if kind == "lit":
        return re.escape(node[1])
    if kind == "set":
        return node[1]
    if kind == "rep":
        _, inner, lo, hi, greedy = node
        body = _rx(inner)
        if inner[0] not in ("set",):
            body = f"(?:{body})"
        quant = f"{{{lo},{'' if hi is None else hi}}}" if lo != hi else f"{{{lo}}}"
        return body + quant + ("" if greedy or lo == hi else "?")
    if kind == "seq":
        return "".join(_rx(n) if n[0] != "alt" else _rx(n) for n in node[1])
    if kind == "alt":
        return "(?:" + "|".join(_rx(n) for n in node[1]) + ")"
    if kind == "case":
        return f"(?{'i' if node[2] else '-i'}:{_rx(node[1])})"
    if kind == "raw":
        return f"(?:{node[1]})"
    if kind == "bound":
        return node[2] if node[3] else node[1]
    if kind == "look":
        _, inner, ahead, negated = node
        if ahead:
            return f"(?{'!' if negated else '='}{_rx(inner)})"
        return _lookbehind(inner, negated)
    if kind == "named":
        return f"(?P<{node[2]}>{_rx(node[1])})"
    raise MatlabError("Invalid pattern.")


def _fixed_variants(node: tuple, budget: list[int], first: bool = True) -> list[str]:
    """Regular expressions of fixed width that together match what ``node`` matches (for look-behind)."""
    kind = node[0]
    if kind in ("lit", "set", "bound"):
        return [_rx(node)]
    if kind == "rep":
        _, inner, lo, hi, greedy = node
        inner_variants = _fixed_variants(inner, budget, False)
        if len(inner_variants) != 1:
            raise MatlabError("lookBehindBoundary: this pattern is too general to look behind for.")
        body = inner_variants[0] if inner[0] == "set" else f"(?:{inner_variants[0]})"
        if hi is None and first:
            counts = [lo]  # looking back, 'at least lo' at the start is the same as 'exactly lo'
        else:
            top = lo + 16 if hi is None else hi
            counts = list(range(lo, top + 1))
        budget[0] -= len(counts)
        if budget[0] < 0:
            raise MatlabError("lookBehindBoundary: this pattern is too general to look behind for.")
        return [f"{body}{{{c}}}" for c in counts]
    if kind == "seq":
        out = [""]
        for k, part in enumerate(node[1]):
            variants = _fixed_variants(part, budget, first and k == 0)
            out = [a + b for a in out for b in variants]
            if len(out) > 400:
                raise MatlabError("lookBehindBoundary: this pattern is too general to look behind for.")
        return out
    if kind == "alt":
        out = []
        for part in node[1]:
            out += _fixed_variants(part, budget, first)
        return out
    if kind == "case":
        flag = "i" if node[2] else "-i"
        return [f"(?{flag}:{v})" for v in _fixed_variants(node[1], budget, first)]
    if kind == "named":
        return _fixed_variants(node[1], budget, first)
    if kind == "look":
        return [_rx(node)]
    raise MatlabError("lookBehindBoundary: a regexpPattern cannot be used inside a look-behind of variable width.")


def _lookbehind(inner: tuple, negated: bool) -> str:
    plain = f"(?<{'!' if negated else '='}{_rx(inner)})"
    try:
        re.compile(plain)
        return plain
    except re.error:
        pass
    variants = _fixed_variants(inner, [200])
    if negated:
        return "".join(f"(?<!{v})" for v in variants)
    return "(?:" + "|".join(f"(?<={v})" for v in variants) + ")"


class Pattern(MObject):
    """A MATLAB pattern object."""

    def __init__(self, node: tuple, label: str):
        self.node = node
        self.label = label  # how the pattern is shown, e.g. digitsPattern + "px"
        self._compiled: dict[bool, Any] = {}

    def class_name(self) -> str:
        return "pattern"

    def isa_class(self, name: str) -> bool:
        return name == "pattern"

    def regex(self) -> str:
        return _rx(self.node)

    def compiled(self, ignore_case: bool = False):
        rx = self._compiled.get(ignore_case)
        if rx is None:
            try:
                rx = re.compile(self.regex(), re.DOTALL | (re.IGNORECASE if ignore_case else 0))
            except re.error as e:
                raise MatlabError(f"This pattern cannot be searched for ({e}).") from None
            self._compiled[ignore_case] = rx
        return rx

    # operators -------------------------------------------------------------------------------
    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("+", "|"):
            return NotImplemented
        a, b = as_pattern(left), as_pattern(right)
        if op == "+":
            parts = (list(a.node[1]) if a.node[0] == "seq" else [a.node]) + \
                    (list(b.node[1]) if b.node[0] == "seq" else [b.node])
            return Pattern(("seq", parts), f"{_wrap(a, '+')} + {_wrap(b, '+')}")
        parts = (list(a.node[1]) if a.node[0] == "alt" else [a.node]) + \
                (list(b.node[1]) if b.node[0] == "alt" else [b.node])
        return Pattern(("alt", parts), f"{a.label} | {b.label}")

    def unary_op(self, interp, op: str) -> Any:
        if op != "~":
            return NotImplemented
        node = self.node
        if node[0] == "bound":
            return Pattern(("bound", node[1], node[2], not node[3]), _negate_label(self.label))
        if node[0] == "look":
            return Pattern(("look", node[1], node[2], not node[3]), _negate_label(self.label))
        raise MatlabError("The ~ operator is only defined for boundary patterns such as letterBoundary, "
                          "textBoundary, lookAheadBoundary and lookBehindBoundary.",
                          "MATLAB:pattern:InvalidNotOperand")

    # display -----------------------------------------------------------------------------------
    def summary_text(self, in_cell: bool) -> str:
        return "1×1 pattern" if in_cell else "[1×1 pattern]"

    def display_lines(self) -> list[str]:
        return ["  pattern", "", "  Matching:", "", "    " + self.label]

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, Pattern) and other.node == self.node


def _wrap(p: Pattern, op: str) -> str:
    return f"({p.label})" if p.node[0] == "alt" and op == "+" else p.label


def _negate_label(label: str) -> str:
    return label[1:] if label.startswith("~") else "~" + label


def _texts_of(v: Any) -> list[str] | None:
    """The texts of a char vector, string array or cellstr; None for other values."""
    if isinstance(v, MString):
        return [str(v)]
    if isinstance(v, StringArray):
        return [x for x in v.values_f() if x is not None]
    if is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return ["".join(r) for r in v]
        return [to_str(v)]
    if isinstance(v, Cell):
        out = []
        for x in v.values_f():
            if not (is_char(x) or isinstance(x, MString)):
                return None
            out.append(to_str(x))
        return out
    return None


def _quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def as_pattern(v: Any) -> Pattern:
    """A pattern for a pattern object or for text (several texts match any one of them)."""
    if isinstance(v, Pattern):
        return v
    texts = _texts_of(v)
    if texts is None:
        raise MatlabError("A pattern must be text or a pattern object, not a value of class "
                          f"'{class_name(v)}'.", "MATLAB:pattern:InvalidInput")
    if len(texts) == 1:
        return Pattern(("lit", texts[0]), _quote(texts[0]))
    # the longest alternative wins when several start at the same place
    ordered = sorted(range(len(texts)), key=lambda k: -len(texts[k]))
    return Pattern(("alt", [("lit", texts[k]) for k in ordered]), " | ".join(_quote(t) for t in texts))


def is_pattern(v: Any) -> bool:
    return isinstance(v, Pattern)


# -- constructors --------------------------------------------------------------------------------------

def _count_args(args: list, name: str, default_lo: int = 1) -> tuple[int, int | None, str]:
    """(lo, hi, label suffix) from name(), name(N) or name(min, max)."""
    if not args:
        return default_lo, None, ""
    vals = []
    for a in args[:2]:
        x = float(np.real(to_numeric(a)).flat[0])
        if np.isinf(x):
            vals.append(None)
        elif x < 0 or x != int(x):
            raise MatlabError(f"{name}: the counts must be nonnegative whole numbers.",
                              "MATLAB:pattern:InvalidCount")
        else:
            vals.append(int(x))
    if len(vals) == 1:
        if vals[0] is None:
            raise MatlabError(f"{name}: the count must be finite.", "MATLAB:pattern:InvalidCount")
        return vals[0], vals[0], f"({vals[0]})"
    lo, hi = vals
    if lo is None or (hi is not None and hi < lo):
        raise MatlabError(f"{name}: the minimum count must not exceed the maximum.", "MATLAB:pattern:InvalidCount")
    return lo, hi, f"({lo},{'Inf' if hi is None else hi})"


def _class_pattern(name: str, one: str):
    def f(interp, args, nargout):
        nargs(args, 0, 2, name)
        lo, hi, suffix = _count_args(list(args), name)
        return Pattern(("rep", ("set", one), lo, hi, True), name + suffix)
    builtin(name)(f)


_class_pattern("digitsPattern", _DIGIT)
_class_pattern("lettersPattern", _LETTER)
_class_pattern("alphanumericsPattern", _ALNUM)
_class_pattern("whitespacePattern", _SPACE)


@builtin("wildcardPattern")
def _wildcard(interp, args, nargout):
    args = list(args)
    one = "(?s:.)"
    label_extra = ""
    for k in range(len(args)):
        if _is_word(args[k], "except") and k + 1 < len(args):
            exc = as_pattern(args[k + 1])
            one = f"(?:(?!{exc.regex()})(?s:.))"
            label_extra = f'"Except",{exc.label}'
            del args[k:k + 2]
            break
    nargs(args, 0, 2, "wildcardPattern")
    lo, hi, suffix = _count_args(args, "wildcardPattern", 0)
    label = "wildcardPattern" + suffix
    if label_extra:
        label = label[:-1] + "," + label_extra + ")" if suffix else f"wildcardPattern({label_extra})"
    return Pattern(("rep", ("raw", one), lo, hi, False), label)


def _is_word(v: Any, word: str) -> bool:
    return (isinstance(v, MString) or (is_char(v) and v.ndim == 2 and v.shape[0] == 1)) and to_str(v).lower() == word


@builtin("characterListPattern")
def _char_list(interp, args, nargout):
    nargs(args, 1, 2, "characterListPattern")
    if len(args) == 2:
        a, b = to_str(args[0]), to_str(args[1])
        if len(a) != 1 or len(b) != 1 or a > b:
            raise MatlabError("characterListPattern(startChar, endChar) needs two single characters in order.",
                              "MATLAB:pattern:InvalidRange")
        return Pattern(("set", f"[{re.escape(a)}-{re.escape(b)}]"), f'characterListPattern("{a}","{b}")')
    chars = to_str(args[0])
    if not chars:
        raise MatlabError("characterListPattern needs at least one character.", "MATLAB:pattern:InvalidInput")
    body = "".join("\\" + c if c in "\\]^-[" else c for c in chars)
    return Pattern(("set", f"[{body}]"), f"characterListPattern({_quote(chars)})")


@builtin("pattern")
def _pattern(interp, args, nargout):
    nargs(args, 0, 1, "pattern")
    if not args:
        return Pattern(("lit", ""), '""')
    return as_pattern(args[0])


@builtin("optionalPattern")
def _optional(interp, args, nargout):
    nargs(args, 1, 1, "optionalPattern")
    p = as_pattern(args[0])
    return Pattern(("rep", p.node, 0, 1, True), f"optionalPattern({p.label})")


def _repeat(name: str, greedy: bool, default_lo: int):
    def f(interp, args, nargout):
        nargs(args, 1, 3, name)
        p = as_pattern(args[0])
        lo, hi = default_lo, None
        if len(args) > 1:
            lo2, hi2, _ = _count_args(list(args[1:]), name)
            lo, hi = (lo2, None) if len(args) == 2 else (lo2, hi2)
        suffix = "" if len(args) == 1 else (f",{lo}" if len(args) == 2 else f",{lo},{'Inf' if hi is None else hi}")
        return Pattern(("rep", p.node, lo, hi, greedy), f"{name}({p.label}{suffix})")
    builtin(name)(f)


_repeat("asManyOfPattern", True, 1)
_repeat("asFewOfPattern", False, 0)


@builtin("caseInsensitivePattern")
def _case_insensitive(interp, args, nargout):
    nargs(args, 1, 1, "caseInsensitivePattern")
    p = as_pattern(args[0])
    return Pattern(("case", p.node, True), f"caseInsensitivePattern({p.label})")


@builtin("caseSensitivePattern")
def _case_sensitive(interp, args, nargout):
    nargs(args, 1, 1, "caseSensitivePattern")
    p = as_pattern(args[0])
    return Pattern(("case", p.node, False), f"caseSensitivePattern({p.label})")


@builtin("regexpPattern")
def _regexp_pattern(interp, args, nargout):
    nargs(args, 1, None, "regexpPattern")
    from .strings import _translate_regex
    expr = to_str(args[0])
    flags = ""
    k = 1
    multiline = False
    while k + 1 < len(args):
        name, value = to_str(args[k]).lower(), args[k + 1]
        truthy = bool(np.real(to_numeric(value)).flat[0]) if not (is_char(value) or isinstance(value, MString)) \
            else to_str(value).lower() in ("true", "on")
        if name == "ignorecase":
            flags += "i" if truthy else ""
        elif name == "multiline" or name == "lineanchors":
            multiline = truthy
            flags += "m" if truthy else ""
        elif name == "dotexceptnewline":
            flags = flags + ("-s" if truthy else "")
        elif name == "freespacing":
            flags += "x" if truthy else ""
        else:
            raise MatlabError(f"regexpPattern: unknown option '{to_str(args[k])}'.", "MATLAB:pattern:InvalidOption")
        k += 2
    translated = _translate_regex(expr, multiline)
    try:
        re.compile(translated)
    except re.error as e:
        raise MatlabError(f"regexpPattern: invalid regular expression '{expr}' ({e}).") from None
    translated = re.sub(r"\((?!\?)", "(?:", translated) if "(?P<" not in translated else translated
    if flags:
        on = "".join(c for c in flags.replace("-s", "") if c in "imx")
        off = "s" if "-s" in flags else ""
        translated = f"(?{on}{'-' + off if off else ''}:{translated})"
    return Pattern(("raw", translated), f"regexpPattern({_quote(expr)})")


def _edge(name: str, start_rx: str, end_rx: str):
    def f(interp, args, nargout):
        nargs(args, 0, 1, name)
        kind = to_str(args[0]).lower() if args else "either"
        if kind not in ("start", "end", "either"):
            raise MatlabError(f"{name}: the type must be 'start', 'end' or 'either'.", "MATLAB:pattern:InvalidOption")
        pos = {"start": start_rx, "end": end_rx, "either": f"(?:{start_rx}|{end_rx})"}[kind]
        neg = f"(?!{pos})"
        return Pattern(("bound", pos, neg, False), name + (f'("{kind}")' if args else ""))
    builtin(name)(f)


_edge("textBoundary", r"\A", r"\Z")
_edge("lineBoundary", r"(?:\A|(?<=\n))", r"(?:\Z|(?=\r?\n))")


def _class_boundary(name: str, one: str):
    """A place where a character of the class meets a character outside it (or the end of the text)."""
    start = f"(?:(?<!{one})(?={one}))"
    end = f"(?:(?<={one})(?!{one}))"

    def f(interp, args, nargout):
        nargs(args, 0, 1, name)
        kind = to_str(args[0]).lower() if args else "either"
        if kind not in ("start", "end", "either"):
            raise MatlabError(f"{name}: the type must be 'start', 'end' or 'either'.", "MATLAB:pattern:InvalidOption")
        pos = {"start": start, "end": end, "either": f"(?:{start}|{end})"}[kind]
        return Pattern(("bound", pos, f"(?!{pos})", False), name + (f'("{kind}")' if args else ""))
    builtin(name)(f)


_class_boundary("letterBoundary", _LETTER)
_class_boundary("digitBoundary", _DIGIT)
_class_boundary("alphanumericBoundary", _ALNUM)
_class_boundary("whitespaceBoundary", _SPACE)


@builtin("lookAheadBoundary")
def _look_ahead(interp, args, nargout):
    nargs(args, 1, 1, "lookAheadBoundary")
    p = as_pattern(args[0])
    return Pattern(("look", p.node, True, False), f"lookAheadBoundary({p.label})")


@builtin("lookBehindBoundary")
def _look_behind(interp, args, nargout):
    nargs(args, 1, 1, "lookBehindBoundary")
    p = as_pattern(args[0])
    pat = Pattern(("look", p.node, False, False), f"lookBehindBoundary({p.label})")
    pat.regex()  # fail now if the pattern cannot be looked behind for
    return pat


@builtin("namedPattern")
def _named(interp, args, nargout):
    nargs(args, 1, 3, "namedPattern")
    p = as_pattern(args[0])
    name = to_str(args[1]) if len(args) > 1 else "pattern"
    if not re.fullmatch(r"[A-Za-z]\w*", name):
        raise MatlabError("namedPattern: the name must be a valid identifier.", "MATLAB:pattern:InvalidName")
    return Pattern(("named", p.node, name), f"namedPattern({p.label},{_quote(name)})")


@builtin("ispattern")
def _ispattern(interp, args, nargout):
    nargs(args, 1, 1, "ispattern")
    return np.array([[isinstance(args[0], Pattern)]])


# -- text arguments ---------------------------------------------------------------------------------

class TextArg:
    """A text argument: its kind ('char', 'cell' or 'string'), shape and items (None = missing)."""

    def __init__(self, v: Any, name: str):
        if isinstance(v, MString):
            self.kind, self.shape, self.items = "string", (1, 1), [str(v)]
        elif isinstance(v, StringArray):
            self.kind, self.shape, self.items = "string", v.data.shape, v.values_f()
        elif is_char(v):
            if v.ndim == 2 and v.shape[0] > 1:
                raise MatlabError(f"{name}: the text must be a string array, a character vector or a cell array "
                                  "of character vectors, not a character matrix.", "MATLAB:string:MustBeStringScalar")
            self.kind, self.shape, self.items = "char", (1, 1), [to_str(v)]
        elif isinstance(v, Cell):
            items = []
            for x in v.values_f():
                if isinstance(x, MString):
                    items.append(str(x))
                elif is_char(x) and (x.size == 0 or (x.ndim == 2 and x.shape[0] == 1)):
                    items.append(to_str(x))
                else:
                    raise MatlabError(f"{name}: every cell must hold a character vector.",
                                      "MATLAB:string:MustBeStringScalar")
            self.kind, self.shape, self.items = "cell", v.data.shape, items
        elif hasattr(v, "categories_list") or class_name(v) == "categorical":
            texts = BUILTINS["cellstr"](None, [v], 1)
            arg = TextArg(texts, name)
            self.kind, self.shape, self.items = "cell", arg.shape, arg.items
        else:
            raise MatlabError(f"{name}: the first argument must be text (a string array, a character vector or "
                              "a cell array of character vectors).", "MATLAB:string:MustBeStringScalar")

    @property
    def scalar(self) -> bool:
        return len(self.items) == 1

    def logical(self, flags: list) -> np.ndarray:
        return np.array(flags, dtype=bool).reshape(self.shape, order="F")

    def numeric(self, values: list) -> np.ndarray:
        return np.array(values, dtype=float).reshape(self.shape, order="F")

    def like(self, items: list, shape: tuple | None = None) -> Any:
        """Text of the same kind as this argument."""
        shape = self.shape if shape is None else shape
        return make_text(self.kind, items, shape)


def make_text(kind: str, items: list, shape: tuple) -> Any:
    n = 1
    for d in shape:
        n *= d
    if kind == "char":
        if n == 1:
            return char(items[0] or "")
        kind = "cell"
    data = np.empty(n, dtype=object)
    for k, x in enumerate(items):
        data[k] = (char(x or "") if kind == "cell" else x)
    data = data.reshape(shape, order="F")
    return Cell(data) if kind == "cell" else string_result(data)


def _options(args: list, names: dict, fname: str) -> tuple[list, dict]:
    """Split trailing Name, Value options (by lower-case name) from the positional arguments."""
    args = list(args)
    opts: dict = {}
    while len(args) >= 2 and (isinstance(args[-2], MString) or is_char(args[-2])) \
            and not isinstance(args[-2], Pattern) and to_str(args[-2]).lower() in names \
            and len(args) - 2 >= names[to_str(args[-2]).lower()]:
        opts[to_str(args[-2]).lower()] = args[-1]
        del args[-2:]
    return args, opts


def _truth(v: Any) -> bool:
    if isinstance(v, MString) or is_char(v):
        return to_str(v).lower() in ("true", "on", "1")
    return bool(np.real(to_numeric(v)).flat[0])


def _search(pat: Any, ignore_case: bool):
    return as_pattern(pat).compiled(ignore_case)


def _spans(rx, s: str) -> list[tuple[int, int]]:
    """Non-overlapping matches from left to right; empty matches only count for zero-width patterns."""
    out = []
    pos = 0
    n = len(s)
    while pos <= n:
        m = rx.search(s, pos)
        if m is None:
            break
        a, b = m.span()
        out.append((a, b))
        pos = b if b > a else b + 1
    return out


# -- yes/no functions ----------------------------------------------------------------------------------

def _predicate(name: str, test, strict: bool = False):
    def f(interp, args, nargout):
        if strict:  # everything after the pattern is a Name,Value pair; an unknown name is an error
            nargs(args, 2, None, name)
            opts = {k.lower(): v for k, v in parse_options(list(args[2:]), ["IgnoreCase"], name,
                                                            first_arg=3).items()}
            args = list(args[:2])
        else:
            args, opts = _options(list(args), {"ignorecase": 2}, name)
            nargs(args, 2, 2, name)
        t = TextArg(args[0], name)
        rx = _search(args[1], _truth(opts.get("ignorecase", False)))
        return t.logical([s is not None and test(rx, s) for s in t.items])
    builtin(name)(f)


_predicate("contains", lambda rx, s: rx.search(s) is not None, strict=True)
_predicate("startsWith", lambda rx, s: rx.match(s) is not None, strict=True)
_predicate("matches", lambda rx, s: rx.fullmatch(s) is not None)


def _ends(rx, s: str) -> bool:
    for start in range(len(s) + 1):
        m = rx.fullmatch(s, start)
        if m is not None:
            return True
    return False


_predicate("endsWith", _ends, strict=True)


@builtin("count")
def _count(interp, args, nargout):
    args, opts = _options(list(args), {"ignorecase": 2}, "count")
    nargs(args, 2, 2, "count")
    t = TextArg(args[0], "count")
    rx = _search(args[1], _truth(opts.get("ignorecase", False)))
    return t.numeric([0 if s is None else sum(1 for a, b in _spans(rx, s) if b > a) for s in t.items])


# -- replace / erase ---------------------------------------------------------------------------------------

def _replace_items(t: TextArg, old: Any, new: Any, name: str) -> list:
    new_texts = _texts_of(new)
    if new_texts is None:
        raise MatlabError(f"{name}: the replacement must be text.", "MATLAB:string:MustBeStringScalar")
    if isinstance(new, StringArray) and any(x is None for x in new.values_f()):
        raise MatlabError(f"{name}: the replacement must not be missing.")
    old_texts = None if isinstance(old, Pattern) else _texts_of(old)
    if old_texts is not None and len(old_texts) > 1 and len(new_texts) == len(old_texts):
        # pairwise: each old text has its own replacement
        order = sorted(range(len(old_texts)), key=lambda k: -len(old_texts[k]))
        rx = re.compile("|".join(f"({re.escape(old_texts[k])})" for k in order), re.DOTALL)
        table = [new_texts[k] for k in order]

        def sub(s: str) -> str:
            return rx.sub(lambda m: table[m.lastindex - 1], s)
        return [None if s is None else sub(s) for s in t.items]
    if len(new_texts) != 1:
        if len(new_texts) == len(t.items):  # one replacement per element of the text
            rx = _search(old, False)
            return [None if s is None else rx.sub(lambda m, r=r: r, s) for s, r in zip(t.items, new_texts)]
        raise MatlabError(f"{name}: the number of replacements must be 1 or match the number of texts to replace.",
                          "MATLAB:string:ReplaceSizeMismatch")
    rx = _search(old, False)
    rep = new_texts[0]
    out = []
    for s in t.items:
        if s is None:
            out.append(None)
            continue
        pieces, last = [], 0
        for a, b in _spans(rx, s):
            if b == a and not _zero_width(old):
                continue
            pieces.append(s[last:a])
            pieces.append(rep)
            last = b
        pieces.append(s[last:])
        out.append("".join(pieces))
    return out


def _zero_width(pat: Any) -> bool:
    return isinstance(pat, Pattern) and pat.node[0] in ("bound", "look")


@builtin("replace")
def _replace(interp, args, nargout):
    nargs(args, 3, 3, "replace")
    t = TextArg(args[0], "replace")
    return t.like(_replace_items(t, args[1], args[2], "replace"))


@builtin("erase")
def _erase(interp, args, nargout):
    nargs(args, 2, 2, "erase")
    t = TextArg(args[0], "erase")
    return t.like(_replace_items(t, args[1], MString(""), "erase"))


# -- extract ---------------------------------------------------------------------------------------------------

def _stack_pieces(t: TextArg, pieces: list[list], name: str) -> Any:
    """Combine the pieces found in each text: N-by-1 for one text, one row (or column) per text otherwise."""
    kind = "cell" if t.kind == "char" else t.kind
    if t.scalar:
        return make_text(kind, pieces[0], (len(pieces[0]), 1))
    counts = {len(p) for p in pieces}
    if len(counts) != 1:
        raise MatlabError(f"{name}: every text must give the same number of pieces. Call {name} on each "
                          "element instead.", "MATLAB:string:ResultSizeMismatch")
    n = counts.pop()
    m = len(pieces)
    if len(t.shape) == 2 and t.shape[0] == 1 and m > 1:  # a row of texts: the pieces go down the columns
        items = [pieces[j][i] for j in range(m) for i in range(n)]
        return make_text(kind, items, (n, m))
    items = [pieces[j][i] for i in range(n) for j in range(m)]
    return make_text(kind, items, (m, n) if len(t.shape) == 2 else tuple(t.shape) + (n,))


@builtin("extract")
def _extract(interp, args, nargout):
    nargs(args, 2, 2, "extract")
    t = TextArg(args[0], "extract")
    rx = _search(args[1], False)
    pieces = [[] if s is None else [s[a:b] for a, b in _spans(rx, s) if b > a] for s in t.items]
    return _stack_pieces(t, pieces, "extract")


def _is_position(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and not is_char(v)


def _positions(v: Any, n: int, name: str) -> list[int]:
    a = np.real(to_numeric(v)).astype(float).flatten(order="F")
    if a.size not in (1, n):
        raise MatlabError(f"{name}: the positions must be a scalar or match the size of the text.",
                          "MATLAB:string:PositionSizeMismatch")
    if np.any(a != np.floor(a)) or np.any(~np.isfinite(a)):
        raise MatlabError(f"{name}: positions must be whole numbers.", "MATLAB:string:PositionMustBeInteger")
    return [int(x) for x in (np.repeat(a, n) if a.size == 1 else a)]


def _missing_for(t: TextArg):
    return None if t.kind == "string" else ""


def _before_after(name: str, before: bool):
    def f(interp, args, nargout):
        nargs(args, 2, 2, name)
        t = TextArg(args[0], name)
        out = []
        if _is_position(args[1]):
            for s, p in zip(t.items, _positions(args[1], len(t.items), name)):
                if s is None:
                    out.append(None)
                    continue
                if before:
                    if p < 1 or p > len(s) + 1:
                        raise MatlabError("Numeric value exceeds the number of characters in element 1.",
                                          "MATLAB:string:PositionOutOfRange")
                    out.append(s[:p - 1])
                else:
                    if p < 0 or p > len(s):
                        raise MatlabError("Numeric value exceeds the number of characters in element 1.",
                                          "MATLAB:string:PositionOutOfRange")
                    out.append(s[p:])
            return t.like(out)
        rx = _search(args[1], False)
        for s in t.items:
            if s is None:
                out.append(None)
                continue
            m = rx.search(s)
            if m is None:
                out.append(_missing_for(t))
            else:
                out.append(s[:m.start()] if before else s[m.end():])
        return t.like(out)
    builtin(name)(f)


_before_after("extractBefore", True)
_before_after("extractAfter", False)


def _between_spans(s: str, start: Any, end: Any, k: int, n: int, name: str, inclusive: bool
                   ) -> list[tuple[int, int]]:
    """Spans of s between the start and end markers (0-based, end exclusive).

    A marker is text or a pattern (every occurrence counts) or a character position.
    """
    start_pos = _positions(start, n, name)[k] if _is_position(start) else None
    end_pos = _positions(end, n, name)[k] if _is_position(end) else None
    for p in (start_pos, end_pos):
        if p is not None and (p < 1 or p > len(s)):
            raise MatlabError("Numeric value exceeds the number of characters in element 1.",
                              "MATLAB:string:PositionOutOfRange")
    start_rx = None if start_pos is not None else _search(start, False)
    end_rx = None if end_pos is not None else _search(end, False)
    out = []
    pos = 0
    while pos <= len(s):
        if start_rx is None:
            sa, sb = start_pos - 1, start_pos
        else:
            m = start_rx.search(s, pos)
            if m is None:
                break
            sa, sb = m.span()
        if end_rx is None:
            ea, eb = end_pos - 1, end_pos
        else:
            m2 = end_rx.search(s, sb)
            if m2 is None:
                break
            ea, eb = m2.span()
        a, b = (sa, eb) if inclusive else (sb, ea)
        if b < a:
            if start_rx is None and end_rx is None:
                raise MatlabError("The start position must not be after the end position.",
                                  "MATLAB:string:PositionOutOfRange")
            b = a
        out.append((a, b))
        if start_rx is None or end_rx is None:
            break
        pos = eb if eb > sa else eb + 1
    return out


def _boundaries(opts: dict, name: str) -> bool | None:
    if "boundaries" not in opts:
        return None
    word = to_str(opts["boundaries"]).lower()
    if word not in ("inclusive", "exclusive"):
        raise MatlabError(f"{name}: 'Boundaries' must be 'inclusive' or 'exclusive'.", "MATLAB:string:InvalidOption")
    return word == "inclusive"


@builtin("extractBetween")
def _extract_between(interp, args, nargout):
    args, opts = _options(list(args), {"boundaries": 3}, "extractBetween")
    nargs(args, 3, 3, "extractBetween")
    t = TextArg(args[0], "extractBetween")
    inclusive = _boundaries(opts, "extractBetween")
    n = len(t.items)
    pieces = []
    for k, s in enumerate(t.items):
        if s is None:
            pieces.append([None])
            continue
        inc = inclusive
        if inc is None:
            inc = _is_position(args[1]) and _is_position(args[2])
        pieces.append([s[a:b] for a, b in _between_spans(s, args[1], args[2], k, n, "extractBetween", inc)])
    if not t.scalar and all(len(p) == 1 for p in pieces):
        kind = "cell" if t.kind == "char" else t.kind
        return make_text(kind, [p[0] for p in pieces], t.shape)
    return _stack_pieces(t, pieces, "extractBetween")


def _replace_between(name: str, erase: bool):
    def f(interp, args, nargout):
        args, opts = _options(list(args), {"boundaries": 3 if erase else 4}, name)
        nargs(args, 3 if erase else 4, 3 if erase else 4, name)
        t = TextArg(args[0], name)
        inclusive = _boundaries(opts, name)
        if erase:
            news = [""] * len(t.items)
        else:
            texts = _texts_of(args[3])
            if texts is None or len(texts) not in (1, len(t.items)):
                raise MatlabError(f"{name}: the new text must be one text or one per element.",
                                  "MATLAB:string:ReplaceSizeMismatch")
            news = texts * len(t.items) if len(texts) == 1 else texts
        n = len(t.items)
        out = []
        for k, s in enumerate(t.items):
            if s is None:
                out.append(None)
                continue
            inc = inclusive
            if inc is None:
                inc = _is_position(args[1]) and _is_position(args[2])
            pieces, last = [], 0
            for a, b in _between_spans(s, args[1], args[2], k, n, name, inc):
                pieces.append(s[last:a])
                pieces.append(news[k])
                last = b
            pieces.append(s[last:])
            out.append("".join(pieces))
        return t.like(out)
    builtin(name)(f)


_replace_between("replaceBetween", False)
_replace_between("eraseBetween", True)


def _insert(name: str, after: bool):
    def f(interp, args, nargout):
        nargs(args, 3, 3, name)
        t = TextArg(args[0], name)
        texts = _texts_of(args[2])
        if texts is None or len(texts) not in (1, len(t.items)):
            raise MatlabError(f"{name}: the text to insert must be one text or one per element.",
                              "MATLAB:string:ReplaceSizeMismatch")
        news = texts * len(t.items) if len(texts) == 1 else texts
        out = []
        if _is_position(args[1]):
            for s, p, new in zip(t.items, _positions(args[1], len(t.items), name), news):
                if s is None:
                    out.append(None)
                    continue
                at = p if after else p - 1
                if at < 0 or at > len(s):
                    raise MatlabError("Numeric value exceeds the number of characters in element 1.",
                                      "MATLAB:string:PositionOutOfRange")
                out.append(s[:at] + new + s[at:])
            return t.like(out)
        rx = _search(args[1], False)
        for s, new in zip(t.items, news):
            if s is None:
                out.append(None)
                continue
            pieces, last = [], 0
            for a, b in _spans(rx, s):
                at = b if after else a
                pieces.append(s[last:at])
                pieces.append(new)
                last = at
            pieces.append(s[last:])
            out.append("".join(pieces))
        return t.like(out)
    builtin(name)(f)


_insert("insertAfter", True)
_insert("insertBefore", False)


# -- split / join ---------------------------------------------------------------------------------------------

def _split_one(s: str, rx) -> tuple[list[str], list[str]]:
    parts, found, last = [], [], 0
    for a, b in _spans(rx, s):
        if b == a and (a == 0 or a == len(s)):
            continue
        parts.append(s[last:a])
        found.append(s[a:b])
        last = b
    parts.append(s[last:])
    return parts, found


def _first_unit_dim(shape: tuple) -> int:
    """The dimension that split fills: the first one of size 1 (a new trailing one if there is none)."""
    if all(d == 1 for d in shape):
        return 0
    for k, d in enumerate(shape):
        if d == 1:
            return k
    return len(shape)


@builtin("split")
def _split(interp, args, nargout):
    nargs(args, 1, 3, "split")
    t = TextArg(args[0], "split")
    if len(args) > 1 and not (isinstance(args[1], np.ndarray) and args[1].size == 0 and not is_char(args[1])):
        rx = _search(args[1], False)
    else:
        rx = re.compile(r"\s")
    if any(s is None for s in t.items):
        raise MatlabError("split: the text must not have missing values.", "MATLAB:string:MissingNotSupported")
    results = [_split_one(s, rx) for s in t.items]
    counts = {len(p) for p, _ in results}
    if len(counts) > 1:
        raise MatlabError("Number of elements in each split string must be the same. Use the split function "
                          "on each string separately.", "MATLAB:string:NumSplitsMismatch")
    n = counts.pop() if counts else 0
    kind = "cell" if t.kind == "char" else t.kind
    if len(args) > 2:
        dim = int(np.real(to_numeric(args[2])).flat[0]) - 1
    else:
        dim = _first_unit_dim(t.shape)
    shape = list(t.shape) + [1] * max(0, dim + 1 - len(t.shape))
    if shape[dim] != 1:
        raise MatlabError("split: the text must have size 1 along the dimension that is split into.",
                          "MATLAB:string:InvalidSplitDim")

    def assemble(pick, count: int):
        out_shape = list(shape)
        out_shape[dim] = count
        data = np.empty(out_shape, dtype=object)
        flat_index = np.arange(len(t.items)).reshape(shape, order="F")
        for idx in np.ndindex(*out_shape):
            src = list(idx)
            src[dim] = 0
            data[idx] = pick(int(flat_index[tuple(src)]), idx[dim])
        while data.ndim > 2 and data.shape[-1] == 1:
            data = data.reshape(data.shape[:-1])
        items = list(data.flatten(order="F"))
        return make_text(kind, items, data.shape)

    parts = assemble(lambda j, i: results[j][0][i], n)
    if nargout > 1:
        return parts, assemble(lambda j, i: results[j][1][i], max(n - 1, 0))
    return parts


@builtin("splitlines")
def _splitlines(interp, args, nargout):
    nargs(args, 1, 1, "splitlines")
    return _split(interp, [args[0], Pattern(("raw", r"\r\n|[\n\r\f\v\x85  ]"), "newline")], 1)


@builtin("join")
def _join(interp, args, nargout):
    nargs(args, 1, 3, "join")
    v = args[0]
    if not isinstance(v, (MString, StringArray, Cell)) and not is_char(v):
        if hasattr(v, "has_method") and v.has_method("join"):
            return v.call_method(interp, "join", list(args), nargout)[0]
        raise MatlabError("join: the first argument must be a string array or a cell array of character vectors.",
                          "MATLAB:string:MustBeStringOrCellstr")
    t = TextArg(v, "join")
    delim: Any = MString(" ")
    dim = None
    rest = list(args[1:])
    if len(rest) == 1 and _is_position(rest[0]):
        dim = int(np.real(to_numeric(rest[0])).flat[0]) - 1
    else:
        if rest:
            delim = rest[0]
        if len(rest) > 1:
            dim = int(np.real(to_numeric(rest[1])).flat[0]) - 1
    shape = list(t.shape)
    if dim is None:
        dim = max((k for k, d in enumerate(shape) if d != 1), default=len(shape) - 1)
    if dim >= len(shape):
        return t.like(t.items)
    grid = np.empty(len(t.items), dtype=object)
    for k, s in enumerate(t.items):
        grid[k] = s
    grid = grid.reshape(shape, order="F")
    n = shape[dim]
    dtexts = _texts_of(delim)
    if dtexts is None:
        raise MatlabError("join: the delimiter must be text.", "MATLAB:string:MustBeStringScalar")
    out_shape = list(shape)
    out_shape[dim] = 1
    if len(dtexts) == 1:
        def delim_at(idx, k):
            return dtexts[0]
    else:
        dshape = list(shape_of_text(delim))
        want = list(shape)
        want[dim] = max(n - 1, 0)
        if len(dtexts) == max(n - 1, 0) and (dshape != want):
            def delim_at(idx, k):
                return dtexts[k]
        elif dshape == want:
            dgrid = np.empty(len(dtexts), dtype=object)
            for k, s in enumerate(dtexts):
                dgrid[k] = s
            dgrid = dgrid.reshape(want, order="F")

            def delim_at(idx, k):
                pos = list(idx)
                pos[dim] = k
                return dgrid[tuple(pos)]
        else:
            raise MatlabError("join: the delimiters must be one text, or have one element fewer than the text "
                              "along the joined dimension.", "MATLAB:string:InvalidDelimiterSize")
    out = np.empty(out_shape, dtype=object)
    for idx in np.ndindex(*out_shape):
        pieces = []
        missing = False
        for k in range(n):
            pos = list(idx)
            pos[dim] = k
            s = grid[tuple(pos)]
            if s is None:
                missing = True
                break
            if k:
                pieces.append(delim_at(idx, k - 1))
            pieces.append(s)
        out[idx] = None if missing else "".join(pieces)
    while out.ndim > 2 and out.shape[-1] == 1:
        out = out.reshape(out.shape[:-1])
    kind = "cell" if t.kind == "char" else t.kind
    return make_text(kind, list(out.flatten(order="F")), out.shape)


def shape_of_text(v: Any) -> tuple:
    if isinstance(v, StringArray):
        return v.data.shape
    if isinstance(v, Cell):
        return v.data.shape
    return (1, 1)


# -- strip ---------------------------------------------------------------------------------------------------------

@builtin("strip")
def _strip(interp, args, nargout):
    nargs(args, 1, 3, "strip")
    t = TextArg(args[0], "strip")
    side = "both"
    what: Any = None
    rest = list(args[1:])
    if rest and not isinstance(rest[0], Pattern) and (isinstance(rest[0], MString) or is_char(rest[0])) \
            and to_str(rest[0]).lower() in ("left", "right", "both"):
        side = to_str(rest.pop(0)).lower()
    if rest:
        what = rest.pop(0)
    if rest:
        raise MatlabError("strip: the side must be 'left', 'right' or 'both'.", "MATLAB:string:InvalidOption")
    if what is None:
        one = r"\s"
    elif isinstance(what, Pattern):
        one = what.regex()
    else:
        texts = _texts_of(what)
        if texts is None or len(texts) not in (1, len(t.items)) or any(len(x) != 1 for x in texts):
            raise MatlabError("strip: the character to strip must be a single character (or a pattern).",
                              "MATLAB:string:MustBeCharacter")
        one = None if len(texts) > 1 else re.escape(texts[0])
    out = []
    for k, s in enumerate(t.items):
        if s is None:
            out.append(None)
            continue
        body = one if one is not None else re.escape(_texts_of(what)[k])
        if side in ("left", "both"):
            s = re.sub(rf"\A(?:{body})+", "", s, flags=re.DOTALL)
        if side in ("right", "both"):
            run = re.compile(rf"(?:{body})+", re.DOTALL)
            for start in range(len(s)):
                if run.fullmatch(s, start):  # the longest run that ends the text
                    s = s[:start]
                    break
        out.append(s)
    return t.like(out)


# -- strfind with patterns, append -----------------------------------------------------------------------------------

def _strfind_text(s: str, args: list):
    """Start indices of the text pattern args[1] in s; strfind(s, pat, 'ForceCellOutput'|'Overlaps', tf)."""
    from .strings import _find_all
    pat = to_str(args[1])
    overlaps = True
    if len(args) == 4 and to_str(args[2]).lower() == "overlaps":
        overlaps = bool(to_numeric(args[3]).flat[0])
    hits = _find_all(s, pat)
    if not overlaps:
        kept, last = [], -1
        for h in hits:
            if h >= last:
                kept.append(h)
                last = h + len(pat)
        hits = kept
    return np.array(hits, dtype=float).reshape(1, -1) + 1.0 if hits else np.zeros((0, 0))


@builtin("strfind")
def _strfind(interp, args, nargout):
    nargs(args, 2, 4, "strfind")
    if len(args) >= 2 and isinstance(args[1], Pattern):
        t = TextArg(args[0], "strfind")
        rx = args[1].compiled(False)

        def starts(s):
            vals = [a + 1.0 for a, b in _spans(rx, s or "")]
            return np.array(vals).reshape(1, -1) if vals else np.zeros((1, 0))
        if t.kind == "char" or (t.kind == "string" and t.scalar):
            return starts(t.items[0])
        data = np.empty(len(t.items), dtype=object)
        for k, s in enumerate(t.items):
            data[k] = starts(s)
        return Cell(data.reshape(t.shape, order="F"))
    v = args[0]
    if isinstance(v, (Cell, StringArray)):
        t = TextArg(v, "strfind")
        data = np.empty(len(t.items), dtype=object)
        for k, s in enumerate(t.items):
            data[k] = _strfind_text(s or "", args)
        return Cell(data.reshape(t.shape, order="F"))
    return _strfind_text(to_str(v), args)


@builtin("append")
def _append(interp, args, nargout):
    """append(s1, ..., sN) joins text element by element; unlike strcat it keeps trailing whitespace."""
    nargs(args, 1, None, "append")
    parts = [TextArg(a, "append") for a in args]
    kind = "string" if any(p.kind == "string" for p in parts) else ("cell" if any(p.kind == "cell" for p in parts)
                                                                    else "char")
    shape = (1, 1)
    for p in parts:
        if not p.scalar:
            if shape != (1, 1) and tuple(p.shape) != tuple(shape):
                raise MatlabError("append: all the inputs must have the same size or be scalar.",
                                  "MATLAB:string:SizeMismatch")
            shape = tuple(p.shape)
    n = 1
    for d in shape:
        n *= d
    out = []
    for k in range(n):
        pieces = [p.items[0] if p.scalar else p.items[k] for p in parts]
        out.append(None if any(x is None for x in pieces) else "".join(pieces))
    return make_text(kind, out, shape)
