"""Differential fuzz tests: the fast paths must give exactly the interpreter's answers.

Each generated program runs in a fresh interpreter with the loop compiler, numexpr fusion and the
numeric fast layer switched on, and again with all three off; the outcomes must be identical (see
fuzz_fastpaths_util).  Every test uses a fixed seed, so a failure prints a program that reproduces
it.  ``MYLAB_FUZZ_N=<n>`` runs n cases per test instead of the quick default.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fuzz_fastpaths_util import (  # noqa: E402
    CLASSES, INT_CLASSES, check_all, count, num, rand_double, value, wrap,
)

from mylab import fuse  # noqa: E402

# -- numeric fast layer: small sizes, every class ---------------------------------------------------

_BINOPS = ["+", "-", ".*", "./", ".\\", ".^", "*", "/", "\\", "<", "<=", ">", ">=", "==", "~=", "&", "|"]
_UNARY = ["-{0}", "~{0}", "{0}'", "{0}.'", "+{0}"]
_ELEM = ["abs", "round", "floor", "ceil", "fix", "exp", "sin", "cos", "tan", "atan", "sinh", "cosh", "tanh",
         "asinh", "expm1", "sqrt", "log", "log2", "log10", "sign", "isnan", "isinf", "double", "single",
         "logical", "real", "imag", "conj"]
_REDUCE = ["sum({0})", "prod({0})", "max({0})", "min({0})", "any({0})", "all({0})", "cumsum({0})", "numel({0})",
           "length({0})", "size({0})", "norm({0})", "sort({0})", "unique({0})", "mean({0})", "sum({0}, 2)",
           "sum({0}, 1)", "prod({0}, 2)", "max({0}, [], 2)", "min({0}, [], 1)", "sum({0}, 'all')",
           "max({0}, [], 'all')", "cumsum({0}, 2)", "sort({0}, 'descend')", "sum({0}, 'omitnan')",
           "max({0}, [], 'omitnan')", "min({0}, [], 'includenan')", "any({0}, 2)", "all({0}, 1)",
           "round({0}, 2)", "round({0}, -1)", "size({0}, 1)", "size({0}, 2)", "sum({0}, 'native')",
           "sum({0}, 'double')", "prod({0}, 'native')", "norm({0}, 1)", "norm({0}, Inf)", "dot({0}, {0})",
           "nnz({0})", "find({0})", "cumprod({0})", "diff({0})", "fliplr({0})", "abs({0})"]
_BINARY_FN = ["mod({0}, {1})", "rem({0}, {1})", "max({0}, {1})", "min({0}, {1})", "ismember({0}, {1})",
              "atan2({0}, {1})", "hypot({0}, {1})", "power({0}, {1})", "dot({0}, {1})", "[{0}, {1}]",
              "[{0}; {1}]", "[{0}, {1}; {1}, {0}]", "isequal({0}, {1})", "xor({0}, {1})", "idivide({0}, {1})"]
_MULTI = ["[m, i] = max({0})", "[m, i] = min({0}, [], 2)", "[s, i] = sort({0})", "[s, i] = sort({0}, 'descend')",
          "[u, i, j] = unique({0})", "[tf, loc] = ismember({0}, {1})", "[q, w] = size({0})"]


def _small_shape(rng):
    n = rng.choice([1, 1, 2, 3, 8])
    return rng.choice([(1, n), (n, 1)]) if n > 1 else (1, 1)


def _small_value(rng, cls=None, shape=None):
    if rng.random() < 0.08:
        return value(rng, cls or rng.choice(CLASSES), rng.choice([(0, 0), (1, 0), (0, 3)]))
    return value(rng, cls or rng.choice(CLASSES), shape or _small_shape(rng))


def _fastnum_program(rng: random.Random) -> str:
    stmts = []
    for _ in range(8):
        cls = rng.choice(["double"] * 6 + CLASSES)
        shape = _small_shape(rng)
        a = _small_value(rng, cls, shape)
        other_cls = cls if rng.random() < 0.6 else rng.choice(["double"] * 3 + CLASSES)
        b = _small_value(rng, other_cls, shape if rng.random() < 0.7 else (1, 1))
        kind = rng.random()
        if kind < 0.35:
            stmts.append(f"{a} {rng.choice(_BINOPS)} {b}")
        elif kind < 0.45:
            stmts.append(rng.choice(_UNARY).format(a))
        elif kind < 0.6:
            stmts.append(f"{rng.choice(_ELEM)}({a})")
        elif kind < 0.8:
            stmts.append(rng.choice(_REDUCE).format(a))
        elif kind < 0.92:
            stmts.append(rng.choice(_BINARY_FN).format(a, b))
        else:  # several outputs: "[m, i] = max(...)"
            lhs, rhs = rng.choice(_MULTI).split(" = ", 1)
            stmts.append((lhs.strip("[]").split(", "), rhs.format(a, b)))
    lines = []
    for k, s in enumerate(stmts):
        target = f"r{k}" if isinstance(s, str) else "[" + ", ".join(f"{o}{k}" for o in s[0]) + "]"
        rhs = s if isinstance(s, str) else s[1]
        lines.append(f"try, {target} = {rhs}; catch err, e{k} = {{err.identifier, err.message}}; end")
    return "\n".join(lines)


def test_fastnum_small_values_match():
    rng = random.Random(20261126)
    check_all(_fastnum_program(rng) for _ in range(count(400)))


def test_fastnum_double_scalars_match():
    """Plain double scalars and tiny vectors: the layer's Python-float arithmetic."""
    rng = random.Random(7)

    def prog():
        stmts = []
        for _ in range(12):
            a = num(rand_double(rng, 0.7))
            b = num(rand_double(rng, 0.7))
            shape = _small_shape(rng)
            v = value(rng, "double", shape, special=0.6)
            w = value(rng, "double", shape, special=0.6)
            stmts.append(rng.choice([
                f"{a} {rng.choice(_BINOPS)} {b}", f"({a}) .^ ({b})", f"{v} {rng.choice(_BINOPS)} {w}",
                f"{v} {rng.choice(_BINOPS)} {a}", f"{a} {rng.choice(_BINOPS)} {v}",
                rng.choice(_REDUCE).format(v), f"{rng.choice(_ELEM)}({a})", f"{rng.choice(_ELEM)}({v})",
                rng.choice(_BINARY_FN).format(v, w), rng.choice(_BINARY_FN).format(a, b),
                f"[{a}, {b}; {b}, {a}]", f"{v}(end)",
            ]))
        return wrap(stmts)

    check_all(prog() for _ in range(count(120)))


def test_fastnum_matrix_ops_match():
    """Matrix products, solves and concatenation of small double matrices."""
    rng = random.Random(11)

    def prog():
        stmts = []
        for _ in range(6):
            n = rng.choice([1, 2, 3])
            a = value(rng, "double", (n, n), special=0.15)
            b = value(rng, "double", (n, rng.choice([1, 2])), special=0.15)
            stmts.append(rng.choice([f"{a} * {b}", f"{a} \\ {b}", f"{b}' / {a}", f"[{a}, {b}]", f"{a}^2",
                                     f"{a}'", f"inv({a})", f"det({a})"]))
        return wrap(stmts)

    check_all(prog() for _ in range(count(60)))


# -- numexpr fusion ------------------------------------------------------------------------------

_FUSE_SHAPES = [(1, 64), (64, 1), (8, 8), (4, 4, 5), (2, 50)]


def _fuse_expr(rng: random.Random, depth: int, names: list[str]) -> str:
    if depth == 0 or rng.random() < 0.25:
        r = rng.random()
        if r < 0.65:
            return rng.choice(names)
        if r < 0.85:
            return num(rand_double(rng, 0.3))
        return rng.choice(["a", "b", "pi", "Inf", "NaN", "-0"])
    r = rng.random()
    left = _fuse_expr(rng, depth - 1, names)
    if r < 0.45:
        op = rng.choice(["+", "-", ".*", "./", "+", "-", ".*", "./", ".\\", ".^"])
        return f"({left} {op} {_fuse_expr(rng, depth - 1, names)})"
    if r < 0.55:
        k = num(rand_double(rng, 0.3))
        return rng.choice([f"({left} * {k})", f"({k} * {left})", f"({left} / {k})", f"({k} / {left})"])
    if r < 0.65:
        return f"({left}) .^ {rng.choice(['0', '1', '2', '3', '4', '-1', '-2', '-3', '7', '0.5', '64', '-64'])}"
    if r < 0.72:
        return f"(-{left})"
    funcs = sorted(fuse._FUNCS) + ["sqrt", "log", "asin", "acos", "floor"]
    return f"{rng.choice(funcs)}({left})"


def _fuse_program(rng: random.Random, n_shape=None) -> str:
    shape = n_shape or rng.choice(_FUSE_SHAPES)
    n = 1
    for s in shape:
        n *= s
    dims = ", ".join(str(s) for s in shape)
    lines = []
    for name in ("x", "z", "w"):
        vals = [num(rand_double(rng, 0.25)) for _ in range(n)]
        lines.append(f"{name} = reshape([{', '.join(vals)}], {dims});")
    lines.append(f"a = {num(rand_double(rng, 0.4))}; b = {num(rand_double(rng, 0.4))};")
    names = ["x", "z", "w", "x", "z", "a"]
    r = rng.random()
    if r < 0.1:  # an operand of another shape (implicit expansion or a size error)
        lines.append(f"c = {value(rng, 'double', (1, shape[1]) if len(shape) == 2 else (1, 1), special=0.3)};")
        names.append("c")
    elif r < 0.2:  # an operand of another class
        cls = rng.choice(["single", "int16", "uint8", "logical", "complex", "complex0"])
        lines.append(f"c = {cls}(x);" if not cls.startswith("complex") else
                     f"c = complex(x, {'0' if cls == 'complex0' else 'z'});")
        names.append("c")
    for k in range(4):
        lines.append(f"try, y{k} = {_fuse_expr(rng, rng.randint(2, 4), names)}; "
                     f"catch err, e{k} = {{err.identifier, err.message}}; end")
    return "\n".join(lines)


@pytest.mark.skipif(not fuse.AVAILABLE, reason="numexpr is not installed")
def test_fused_expressions_match(monkeypatch):
    monkeypatch.setattr(fuse, "MIN_SIZE", 40)  # small arrays keep the test fast; the code path is the same
    rng = random.Random(4242)
    check_all(_fuse_program(rng) for _ in range(count(400)))


@pytest.mark.skipif(not fuse.AVAILABLE, reason="numexpr is not installed")
def test_fused_expressions_match_at_the_real_threshold():
    rng = random.Random(99)
    side = int(fuse.MIN_SIZE ** 0.5) + 1

    def prog():
        k = rng.choice([2, 3, -2, 0])
        f = rng.choice(sorted(fuse._FUNCS))
        g = rng.choice(sorted(fuse._FUNCS))
        return (f"x = reshape(linspace(-30, 30, {side * side}), {side}, {side}); x(1:7) = [NaN Inf -Inf -0 0 "
                f"5e-324 1e308]; z = cos(x) * 1e3; a = {num(rand_double(rng, 0.5))};\n"
                f"y1 = {f}(x) .* z - a ./ x + x.^{k};\ny2 = {g}(z ./ x) - 2*x + {f}(-x) .* a;")

    check_all(prog() for _ in range(count(4)))


# -- loop compiler ---------------------------------------------------------------------------------

_USER_FUNCS = {
    "fz_sq.m": "function y = fz_sq(x)\n  y = x.^2 - 1;\nend\n",
    "fz_fact.m": "function r = fz_fact(n)\n  if n <= 1\n    r = 1;\n  else\n    r = n * fz_fact(n - 1);\n  end\nend\n",
    "fz_fib.m": "function r = fz_fib(n)\n  if n < 2\n    r = n;\n  else\n    r = fz_fib(n - 1) + fz_fib(n - 2);\n"
                "  end\nend\n",
    "fz_chk.m": "function y = fz_chk(x)\n  if x > 6\n    error('fz:big', 'Value %g is too big', x);\n  end\n"
                "  y = x / 2;\nend\n",
    "fz_two.m": "function y = fz_two(a, b)\n  y = a * 3 - b;\n  if y < 0\n    y = -y;\n  end\nend\n",
    "fz_sum.m": "function s = fz_sum(v)\n  s = 0;\n  for i = 1:numel(v)\n    s = s + v(i);\n  end\nend\n",
}


@pytest.fixture(scope="module")
def fz_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("fuzzfns")
    for name, src in _USER_FUNCS.items():
        (d / name).write_text(src)
    return [str(d)]


_VSHAPES = [(1, 6), (6, 1), (1, 8), (2, 4)]


class _LoopGen:
    """Random loops made of the statements and expressions the loop compiler handles."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def atom(self, loopvars):
        rng = self.rng
        r = rng.random()
        if r < 0.3:
            return rng.choice(loopvars)
        if r < 0.5:
            return rng.choice(["a", "b", "s", "t"])
        if r < 0.65:
            return num(rand_double(rng, 0.35))
        if r < 0.85:
            k = rng.choice(loopvars)
            return rng.choice([f"v({k})", "v(end)", "v(end - 1)", f"v(mod({k}, numel(v)) + 1)", f"v({k} + 1)",
                               f"v({k} / 2)", f"v(1, {k})", f"M({k}, 1)", "M(end, end)", f"M({k})", "numel(v)",
                               "w(end)", "st.p", "st.q(1)", "c{1}"])
        return rng.choice(["pi", "eps", "Inf", "NaN", "true", "false", "-0", "int8(100)", "uint8(3)", "single(2.5)",
                           "1i", "realmax", "realmin"])

    def expr(self, depth, loopvars):
        rng = self.rng
        if depth == 0 or rng.random() < 0.3:
            return self.atom(loopvars)
        r = rng.random()
        if r < 0.45:
            op = rng.choice(["+", "-", "*", "/", "^", ".^", ".*", "./", "<", ">", "==", "~=", "<=", "&", "|", "\\"])
            return f"({self.expr(depth - 1, loopvars)} {op} {self.expr(depth - 1, loopvars)})"
        if r < 0.55:
            return f"(-{self.expr(depth - 1, loopvars)})"
        if r < 0.6:
            return f"(~{self.expr(depth - 1, loopvars)})"
        if r < 0.85:
            f1 = ["sin", "cos", "tan", "atan", "sinh", "cosh", "tanh", "exp", "sqrt", "log", "log2", "log10", "abs",
                  "floor", "ceil", "fix", "round", "sign", "f", "fz_sq", "fz_chk", "fz_fact", "double", "int8",
                  "uint8", "single", "logical", "isnan", "fz_fib"]
            name = rng.choice(f1)
            arg = self.expr(depth - 1, loopvars)
            if name in ("fz_fact", "fz_fib"):
                arg = f"min(abs(fix({arg})), 9)"
            return f"{name}({arg})"
        f2 = ["mod", "rem", "min", "max", "atan2", "hypot", "fz_two", "g", "power", "idivide"]
        return f"{rng.choice(f2)}({self.expr(depth - 1, loopvars)}, {self.expr(depth - 1, loopvars)})"

    def target(self, loopvars):
        rng = self.rng
        k = rng.choice(loopvars)
        return rng.choice(["s", "t", "a", "b", f"w({k})", f"w({k})", f"w({k} + 2)", "w(end + 1)", f"v({k})",
                           f"M({k}, 2)", f"M(2, {k})", f"M({k})", "st.p", f"st.q({k})", f"c{{{k}}}", "c{end + 1}",
                           f"w({k} / 2)", "w(0)", f"w(2 * {k})", "u"])

    def stmts(self, depth, loopvars, count_):
        rng = self.rng
        out = []
        for _ in range(count_):
            r = rng.random()
            if r < 0.55 or depth == 0:
                out.append(f"{self.target(loopvars)} = {self.expr(rng.randint(1, 3), loopvars)};")
            elif r < 0.75:
                parts = [f"if {self.expr(2, loopvars)}", *self.stmts(depth - 1, loopvars, rng.randint(1, 2))]
                if rng.random() < 0.5:
                    parts += [f"elseif {self.expr(2, loopvars)}", *self.stmts(depth - 1, loopvars, 1)]
                if rng.random() < 0.5:
                    parts += ["else", *self.stmts(depth - 1, loopvars, 1)]
                out.append("\n".join(parts + ["end"]))
            elif r < 0.85:
                out.append(f"if {self.expr(1, loopvars)} {rng.choice(['>', '<', '=='])} "
                           f"{num(rand_double(rng, 0.2))}, {rng.choice(['break', 'continue'])}; end")
            elif len(loopvars) < 3:
                j = "ij"[len(loopvars) - 1]
                out.append(f"for {j} = {self.range_()}\n" + "\n".join(self.stmts(depth - 1, loopvars + [j], 2))
                           + "\nend")
            else:
                out.append(f"s = s + {self.expr(1, loopvars)};")
        return out

    def range_(self):
        rng = self.rng
        return rng.choice(["1:5", "1:8", "2:2:12", "6:-1:1", "1:0.5:4", "0:4", "-2:3", "1:numel(v)", "1:size(M, 1)",
                           "1:0", "3:7"])

    def program(self):
        rng = self.rng
        init = [
            f"a = {rng.choice([num(rand_double(rng, 0.3)), value(rng, rng.choice(CLASSES), (1, 1), 0.3)])};",
            f"b = {num(rand_double(rng, 0.4))}; s = 0; t = {num(rand_double(rng, 0.3))};",
            f"v = {value(rng, rng.choice(['double'] * 5 + CLASSES), rng.choice(_VSHAPES), 0.3)};",
            f"w = {rng.choice(['zeros(1, 10)', '[]', 'zeros(10, 1)', 'ones(1, 3)', 'v'])};",
            f"M = {value(rng, rng.choice(['double', 'double', 'int16', 'single', 'logical']), (4, 3), 0.2)};",
            "st.p = 1; st.q = zeros(1, 4); c = {};",
            f"f = @(x) {rng.choice(['x.^2 - 1', 'sqrt(x)', 'x / 3 + b', 'max(x, 0)', '2 * x'])};",
            f"g = @(x, y) {rng.choice(['x - y', 'x * y + 1', 'mod(x, y)', 'atan2(x, y)'])};",
        ]
        head = rng.choice([f"for k = {self.range_()}", f"for k = {self.range_()}",
                           f"k = 0;\nwhile k < {rng.randint(3, 12)}\nk = k + 1;"])
        body = self.stmts(2, ["k"], rng.randint(2, 5))
        return "\n".join(init + [head] + body + ["end"])


def test_compiled_loops_match(fz_path):
    rng = random.Random(31337)
    gen = _LoopGen(rng)
    check_all((gen.program() for _ in range(count(800))), fz_path)


@pytest.mark.parametrize("cls", ["double", "single", "logical", "char", "complex0"] + INT_CLASSES)
def test_compiled_loops_with_every_class(cls, fz_path):
    rng = random.Random(sum(map(ord, cls)))
    progs = []
    for _ in range(count(12)):
        v = value(rng, cls, rng.choice([(1, 5), (5, 1), (2, 3)]), 0.3)
        a = value(rng, cls, (1, 1), 0.3)
        progs.append(rng.choice([
            f"v = {v}; a = {a}; s = a; w = v; for k = 1:numel(v), s = s + v(k); w(k) = v(k) * a; end",
            f"v = {v}; a = {a}; for k = 1:4, v(k + 1) = v(k) - a; end",
            f"v = {v}; a = {a}; s = 0; for k = 1:numel(v), if v(k) > a, s = s + 1; "
            f"elseif v(k) == a, s = s - 1; end, end",
            f"v = {v}; a = {a}; w = []; for k = 1:7, w(end + 1) = a / k; end",
            f"v = {v}; a = {a}; k = 1; while k <= numel(v) && v(k) ~= a, k = k + 1; end",
            f"v = {v}; a = {a}; m = v(1); for k = 2:numel(v), m = max(m, v(k)); end",
            f"v = {v}; a = {a}; for k = 1:6, a = a * 2 + 1; end",
            f"v = {v}; a = {a}; for k = 1:6, a = a / 0; v(k) = -a; end",
        ]))
    check_all(progs, fz_path)


@pytest.mark.parametrize("code", [
    "s = 0; for k = 1:10, s = s + sqrt(k - 5); end",               # turns complex
    "x = int8(100); for k = 1:6, x = x + 20; end",                 # saturates
    "x = uint8(5); for k = 1:6, x = x - 3; end",
    "x = 1; for k = 1:6, x = x / (k - 3); end",                    # division by zero
    "x = -1; for k = 1:6, x = x / (k - 3); end",
    "v = 1:5; s = 0; for k = 1:8, s = s + v(k); end",              # out of range read
    "v = 1:5; s = 0; for k = 1:8, s = s + v(k / 2); end",          # non-integer index
    "v = 1:5; for k = 1:8, v(k - 1) = k; end",                     # zero index write
    "w = []; for k = 1:6, w(2 * k) = k; end",                      # growth with gaps
    "M = zeros(2); for k = 1:5, M(k, k + 1) = k; end",             # growth of a matrix
    "v = zeros(1, 4); for k = 1:6, v(k) = 1 ./ (k - 3) * -0; end",
    "x = 0; for k = 1:6, x = 1 / (-0 * k); end",                   # -Inf from -0
    "x = 0; for k = 1:5, x = x + 1i; end",
    "x = 0; for k = 1:5, x = mod(-0, k) + rem(-0, k); end",
    "x = 0; for k = 1:5, x = round(-0.4) + fix(-0.5) + ceil(-0.5); end",
    "s = struct('a', 1); for k = 1:5, s.a = s.a * k; s.b(k) = k; end",
    "n = 0; for k = 1:20, if k > 10, break; elseif mod(k, 2), continue; end, n = n + k; end",
    "f = @(x) x^2; s = 0; for k = 1:6, s = s + f(k - 3); end",
    "s = 0; for k = 1:6, s = s + fz_fact(k) - fz_fib(k); end",
    "s = 0; for k = 1:9, s = s + fz_chk(k); end",
    "s = 0; for k = 1:6, s = fz_sum([s k 1]); end",
    "x = 0; for k = 1:5, x = log(k - 3); end",
    "x = 0; for k = 1:5, x = (k - 3) ^ 0.5; end",
    "x = 0; for k = 1:5, x = (-8) ^ (1 / 3) + k; end",
    "x = 0; for k = 1:5, x = atan2(-0, -k) + hypot(k, 1e308); end",
    "x = 0; for k = 1:5, x = exp(700 + k); end",
    "x = 0; for k = 1:5, x = 2 ^ (1020 + k); end",
    "x = 0; for k = 1:5, x = max(NaN, k) + min(k, NaN); end",
    "x = 0; for k = 1:5, if NaN, x = 1; end, end",
    "x = 5; while x, x = x - 1; end",
    "x = 0; for k = 1:5, x = x + int32(7) / int32(2); end",
    "x = 0; for k = 1:5, x = sign(-0) + abs(-0); end",
    "c = {}; for k = 1:5, c{end + 1} = k * 2; end",
    "v = [1 2 3]; for k = 1:5, v(end) = v(end) + k; end",
])
def test_compiled_loop_corner_cases(code, fz_path):
    check_all([code], fz_path)


def test_compiled_functions_match(fz_path):
    """Whole user functions compiled after repeated calls, with every kind of argument."""
    rng = random.Random(5)
    progs = []
    for _ in range(count(40)):
        args = [value(rng, rng.choice(["double"] * 4 + CLASSES), rng.choice([(1, 1), (1, 1), (1, 3)]), 0.4)
                for _ in range(2)]
        call = rng.choice([f"fz_sq({args[0]})", f"fz_two({args[0]}, {args[1]})", f"fz_sum({args[0]})",
                           f"fz_chk({args[0]})", f"fz_fact({args[0]})"])
        progs.append(f"r = {{}}; for k = 1:5, try, r{{k}} = {call}; catch err, r{{k}} = err.message; end, end")
    check_all(progs, fz_path)
