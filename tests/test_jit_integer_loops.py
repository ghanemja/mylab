"""Loops over integer-class ranges in the loop compiler: the results must be the interpreter's.

The loop variable of ``for k = int32(1):n`` keeps its class; arithmetic with a double rounds and
saturates in that class, arithmetic with another integer class is an error, ``double(k)`` converts.
Each program runs with every fast path on and again with all of them off (fuzz_fastpaths_util);
the workspaces (classes and bits), outputs and errors must be identical.  The randomized tests use
fixed seeds; ``MYLAB_FUZZ_N=<n>`` runs n cases per test.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fuzz_fastpaths_util import INT_CLASSES, check_all, count, num, rand_double  # noqa: E402

from mylab import Interpreter, jit  # noqa: E402
from mylab.parser import parse  # noqa: E402

_FUNCS = {
    "fi_twice.m": "function y = fi_twice(x)\n  y = 2 * x - 1;\nend\n",
    "fi_pick.m": "function y = fi_pick(a, b)\n  if a > b\n    y = a;\n  else\n    y = b;\n  end\nend\n",
    "fi_conv.m": "function y = fi_conv(x)\n  y = int16(x) + 3;\nend\n",
    "fi_isum.m": "function s = fi_isum(n)\n  s = 0;\n  for k = int32(1):n\n    s = s + double(k) * 2;\n  end\nend\n",
    "fi_imix.m": "function s = fi_imix(n)\n  s = int8(0);\n  for k = int8(n):-1:int8(-3)\n    s = s + k;\n  end\n"
                 "  for j = uint64(1):3\n    s = s + int8(j);\n  end\nend\n",
}


@pytest.fixture(scope="module")
def fi_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("intloopfns")
    for name, src in _FUNCS.items():
        (d / name).write_text(src)
    return [str(d)]


def _loop_runs_compiled(code: str, path=None) -> list:
    """(compiled?, bail-outs) for each top-level for loop of a program run once."""
    interp = Interpreter(out=lambda s: None, path=list(path or []))
    program = parse(code)
    interp.run_program(program, interp.base)
    out = []
    for stmt in program.body:
        info = stmt.__dict__.get("_jit_static")
        if info is not None and type(stmt).__name__ == "For":
            out.append((any(v is not None for v in info["compiled"].values()), info["bails"]))
    return out


# -- fixed cases ---------------------------------------------------------------------------------

CASES = [
    "s = 0; n = int32(200); for i = int32(1):n, s = s + double(i); end",
    "x = int8(0); for k = int8(120):int8(127), x = k; end",                 # the range ends at intmax
    "x = int8(0); for k = int8(-128):int8(-120), x = k; end",
    "x = uint8(0); n = 0; for k = uint8(250):uint8(255), x = k; n = n + 1; end",
    "t = int16(0); for k = int16(1):int16(300), t = t + k; end",             # saturates at 32767
    "t = uint8(0); c = 0; for k = uint8(10):-1:0, t = t + k * 0.5; c = c + 1; end",  # negative step, halves
    "s = 0; for k = int8(5):-2:int8(-5), s = s + double(k) * 10; end",
    "s = 0; for k = int8(1):0.5:int8(5), s = s + double(k); end",           # values round: 1 2 2 3 3 ...
    "s = 0; for k = int8(120):200, s = s + double(k); end",                 # doubles past intmax saturate
    "s = 0; for k = uint8(3):-1:-4, s = s + double(k); end",                # ... and below zero
    "s = 0; for k = int8(5):int8(1), s = s + double(k); end",               # empty
    "s = 1; for k = int16(4):int16(3), s = 2; end",
    "s = 0; for k = int32(1):int16(5), s = s + double(k); end",             # mixed classes: an error
    "s = 0; for k = int32(1):single(5), s = s + double(k); end",
    "v = zeros(1, 10); for k = int32(1):10, v(k) = double(k) * 2; end",     # integer index
    "v = zeros(1, 10); for k = int32(1):10, v(k) = k * 2; end",
    "M = zeros(3, 4); for k = uint16(1):4, M(2, k) = 7; M(k) = -1; end",
    "c = 0; v = 1:10; for k = int32(1):10, c = c + v(k); end",
    "c = 0; v = 1:4; for k = int32(1):10, c = c + v(k); end",              # out of range: an error
    "c = 0; v = 1:4; for k = int32(-1):3, c = c + v(k); end",
    "c = {}; for k = int32(1):6, c{k} = k; end",
    "c = {}; for k = int64(1):6, c{end + 1} = -k; end",
    "s = int32(0); for k = int32(1):10, s = s + k / 3; end",                # division rounds
    "s = int32(0); for k = int32(1):10, s = s + k / int32(4); end",
    "s = 0; for k = int32(-2):5, s = 10 / k; end",                          # division by zero
    "s = 0; for k = int8(-3):3, s = k / 0; t = -k / 0; u = k / -0; end",
    "s = int64(0); for k = int64(1):10, s = s + k * 3 - 1; end",
    "s = int64(0); for k = int64(1):10, s = s + k / 3; end",
    "s = uint64(0); for k = uint64(1):10, s = s - k; end",                  # saturates at 0
    "s = 0; for k = int64(1):5, s = k * 0.5; t = k + 0.5; u = 2.5 - k; end",
    "s = 0; for k = uint64(1):5, s = k - 2.5; t = k * 1e300; u = k * -1e300; end",
    "s = int64(0); for k = int64(1):5, s = s + int64(9007199254740993); end",  # exact literal
    "s = uint64(0); for k = uint64(1):5, s = s + uint64(18446744073709551615) - k; end",
    "s = int64(9223372036854775000); for k = int64(1):6, s = s + k * 100; end",
    "s = 0; for k = int64(9007199254740990):int64(9007199254740995), s = s + 1; end",
    "s = int8(0); for k = int8(1):10, s = s + int16(k); end",               # mixed classes: an error
    "s = int8(0); for k = int8(1):10, s = k + int16(1); end",
    "s = 0; for k = int8(1):10, s = k == int16(1); end",                    # comparing classes is allowed
    "s = 0; for k = int8(-128):int8(127), s = s + double(-k); end",         # -intmin is intmax
    "s = 0; for k = uint8(0):uint8(255), s = s + double(-k); end",          # unsigned negation is 0
    "s = 0; for k = int32(1):5, s = -k; t = ~k; u = +k; w = k'; z = k.'; end",
    "c = 0; for k = int32(1):20, if mod(k, 3) == 0, c = c + 1; end, end",
    "c = 0; for k = int32(1):20, if k > 5 && k < 15, c = c + 1; end, end",
    "c = 0; for k = int16(0):6, if k, c = c + 1; end, if ~k || k == 3, c = c - 1; end, end",
    "c = int32(0); for k = int32(1):20, c = max(c, k * 2); end",
    "s = 0; for k = int32(1):5, s = s + k; end",                            # s turns int32
    "s = 0; for k = int32(1):5, s = s ^ 2 + k; end",
    "s = 0; for k = int32(1):5, s = k ^ 2; t = 2 ^ k; u = k .^ 0.5; end",
    "s = 0; for k = int32(1):5, s = k \\ 10; t = k .\\ int32(7); end",
    "s = 0; for k = int32(1):5, s = sin(k); end",                           # not defined for integers
    "s = 0; for k = int32(1):5, s = abs(k - 3) + sign(k - 2); end",
    "s = 0; for k = int32(1):5, s = numel(k) + length(k) + size(k, 1); end",
    "s = 0; for k = int32(1):5, s = int8(k * 50); t = uint8(-k); u = int64(k); end",
    "s = 0; for k = int32(1):5, s = uint8(double(k) * -1.5); t = int16(k > 2); u = int32(NaN); end",
    "s = 0; for k = int32(1):5, s = [k, 2]; t = [k; int8(2)]; u = [k]; end",
    "s = 0; for k = int32(1):5, s = s + bitand(k, 3); end",
    "s = 0; for k = int32(1):6, s = s + idivide(k, int32(4)); end",
    "s = 0; for k = int32(1):6, s = s + rem(k, 4) + mod(k, 2.5); end",
    "s = 0; for k = 0x1:0x5, s = s + double(k); end",                       # hexadecimal literals
    "s = 0; for k = int32(1):5, s = s + 0x10; t = k + 0b11s32; end",
    "s = 0; for k = int32(1):5, s = k == true; t = k < 2.5; u = k >= int32(3); w = k ~= NaN; end",
    "s = 0; for k = int32(1):5, s = s + 0.5; end, k",
    "s = 0; for k = int32(1):5, s = s + 1; k = 7; end",                     # the variable is reassigned
    "s = 0; for k = int32(1):5, k = k + 1; s = s + double(k); end",
    "s = 0; for k = int32(2147483640):int32(2147483647), s = s + double(k + 1); end",
    "s = 0; for k = uint32(1):10, s = s + double(k - 5); end",
    "x = int32(0); for k = 1:10, x = x + int32(k); end",                    # integer scalars in double loops
    "s = int32(7); for k = 1:5, s = s * 3; end",
    "s = int32(7); t = 0; for k = 1:5, t = t + s; end",
    "s = uint8(200); for k = 1:6, s = s - 30.5; end",
    "s = int16(5); b = true; for k = 1:6, s = s + b; b = ~b; end",
    "i = int32(5); s = 0; for i = 1:200, s = s + i; end",                   # the old value of k is unused
    "i = 'abc'; s = 0; for i = 1:200, s = s + i; end",
    "f = @(x) x * 2; s = 0; for k = int32(1):5, s = s + f(double(k)); end",
    "f = @(x) x * 2; s = 0; for k = int32(1):5, s = f(k); end",
    "s = 0; for k = int32(1):6, s = s + fi_twice(k); end",
    "s = 0; for k = int32(1):6, s = fi_pick(k, 3) + fi_pick(2.5, k); end",
    "s = 0; for k = int32(1):6, s = fi_conv(k); end",
    "s = 0; for k = int32(1):6, s = fi_twice(int8(k)) + fi_twice(k); end",  # mixed classes: an error
    "n = int32(0); for k = int32(1):10, if k > 6, break; elseif mod(k, 2), continue; end, n = n + k; end",
    # integer ranges inside compiled functions
    "r = zeros(1, 6); for j = 1:6, r(j) = fi_isum(int32(j * 3)); end",
    "r = zeros(1, 6); for j = 1:6, r(j) = fi_isum(j + 0.5); end",
    "r = zeros(1, 6); for j = 1:6, r(j) = fi_isum(int16(j)); end",          # two integer classes: an error
    "r = int8(zeros(1, 6)); for j = 1:6, r(j) = fi_imix(100 + j * 5); end",
    "s = 0; for j = 1:6, s = s + double(fi_imix(-5)); end",                  # an empty range
]


@pytest.mark.parametrize("code", CASES)
def test_integer_loops_match_the_interpreter(code, fi_path):
    check_all([code], fi_path)


@pytest.mark.parametrize("code", [
    "s = 0; n = int32(200000); for i = int32(1):n, s = s + double(i); end",
    "c = 0; for k = int32(1):20, if mod(k, 3) == 0, c = c + 1; end, end",
    "x = int8(0); for k = int8(120):int8(127), x = k; end",
    "s = int64(0); for k = int64(1):10, s = s + k * 3 - 1; end",
    "s = uint8(200); for k = 1:6, s = s - 30.5; end",
    "i = int32(5); s = 0; for i = 1:20, s = s + i; end",
])
def test_integer_loops_are_compiled(code):
    assert _loop_runs_compiled(code) == [(True, 0)]


def test_integer_loop_speed():
    import time
    interp = Interpreter(out=lambda s: None)
    t = time.perf_counter()
    interp.execute("s = 0; n = int32(200000); for i = int32(1):n, s = s + double(i); end")
    assert time.perf_counter() - t < 0.6  # about 0.04 s compiled, 1.2 s interpreted
    assert interp.get_var("s").item() == 200000 * 200001 / 2
    assert interp.get_var("i").dtype == np.int32 and interp.get_var("i").item() == 200000


def test_integer_helpers_round_like_saturate_int():
    from mylab.values import saturate_int
    rng = random.Random(77)
    values = [0.5, -0.5, 1.5, -1.5, 2.5, 0.49999999999999994, -0.49999999999999994, 126.5, 127.5, -128.5,
              -129.0, 255.5, 256.0, 65535.5, 2147483647.5, -2147483648.5, 4294967295.5, 2.0 ** 53, 2.0 ** 63,
              -(2.0 ** 63), 2.0 ** 64, 2.0 ** 63 - 1024, 2.0 ** 64 - 2048, 1e300, -1e300, float("inf"),
              float("-inf"), float("nan"), -0.0, 0.0, 5e-324]
    values += [rng.uniform(-1, 1) * 10.0 ** rng.randint(0, 20) for _ in range(300)]
    for name in INT_CLASSES:
        sat = jit._NS["_Qs" + jit._ICODE[name]]
        for x in values:
            want = saturate_int(np.array([[x]]), name).item()
            want_array = saturate_int(np.array([[x, 0.0]]), name)[0, 0].item()  # the array code too
            got = sat(x)
            assert type(got) is int and got == want == want_array, (name, x, got, want, want_array)


# -- randomized ------------------------------------------------------------------------------------

_SPECIAL_D = ["0.5", "-0.5", "2.5", "1.5", "-2.5", "3", "-7", "0", "-0", "1e300", "-1e300", "NaN", "Inf", "-Inf",
              "0.49999999999999994", "100", "-200", "1000"]


class _IntLoopGen:
    """Random loops over integer-class ranges, made of what the compiler handles and what it must decline."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def bounds(self, cls: str) -> tuple[int, int]:
        """Two ends at most a dozen apart, often next to (or past) the limits of the class."""
        rng = self.rng
        info = np.iinfo(cls)
        r = rng.random()
        if r < 0.45:
            a = rng.randint(max(int(info.min), -20), min(int(info.max), 20))
        elif r < 0.8:  # next to the limits of the class
            a = rng.choice([int(info.min), int(info.max)]) + rng.randint(-6, 6)
            if cls in ("int64", "uint64") and rng.random() < 0.6:
                a = rng.randint(-20, 20)  # mostly in the range the compiler handles
        else:
            a = rng.randint(-300, 300)
        return a, a + rng.randint(-12, 12)

    def end(self, cls: str, v: int, typed: bool) -> str:
        info = np.iinfo(cls)
        if typed:
            return f"{cls}({min(max(v, int(info.min)), int(info.max))})"
        return str(v)  # a double operand (may lie outside the class)

    def range_(self, cls: str) -> str:
        rng = self.rng
        lo, hi = self.bounds(cls)
        ta = rng.random() < 0.7
        tb = not ta or rng.random() < 0.6
        a, b = self.end(cls, lo, ta), self.end(cls, hi, tb)
        r = rng.random()
        if r < 0.5:
            return f"{a}:{b}"
        step = rng.choice(["1", "-1", "2", "-3", "0.5", "-0.5", "1.5", "7", f"{cls}(2)", f"{cls}(-1)", "0", "NaN"])
        return f"{a}:{step}:{b}"

    def operand(self, cls: str) -> str:
        rng = self.rng
        r = rng.random()
        if r < 0.35:
            return "k"
        if r < 0.55:
            return rng.choice(_SPECIAL_D)
        if r < 0.65:
            return f"{cls}({rng.randint(-5, 5)})"
        if r < 0.72:
            return rng.choice(["t", "s", "d"])
        if r < 0.77:
            return rng.choice(["true", "false"])
        if r < 0.82:
            return f"{rng.choice(INT_CLASSES)}(3)"  # usually another class: an error
        if r < 0.9:
            return rng.choice(["v(k)", "v(mod(k, 4) + 1)", "c{1}", "double(k)", "-k", "abs(k)"])
        return num(rand_double(rng, 0.4))

    def expr(self, cls: str, depth: int) -> str:
        rng = self.rng
        if depth == 0 or rng.random() < 0.3:
            return self.operand(cls)
        r = rng.random()
        if r < 0.5:
            op = rng.choice(["+", "-", "*", ".*", "/", "./", "+", "-", "*", "/", "^", ".^", "\\", "==", "<", ">=",
                             "~=", "&", "|"])
            return f"({self.expr(cls, depth - 1)} {op} {self.expr(cls, depth - 1)})"
        if r < 0.6:
            return f"(-{self.expr(cls, depth - 1)})"
        if r < 0.65:
            return f"(~{self.expr(cls, depth - 1)})"
        if r < 0.85:
            f = rng.choice(["double", "abs", "sign", "round", "floor", "mod({0}, 3)", "rem({0}, -4)", "max({0}, 2)",
                            "min({0}, k)", "int8", "uint8", "int16", "int32", "uint32", "int64", cls, cls,
                            "numel", "fi_twice", "fi_conv", "logical", "sqrt"])
            inner = self.expr(cls, depth - 1)
            return f.format(inner) if "{0}" in f else f"{f}({inner})"
        return f"fi_pick({self.expr(cls, depth - 1)}, {self.expr(cls, depth - 1)})"

    def stmts(self, cls: str, count_: int) -> list[str]:
        rng = self.rng
        out = []
        for _ in range(count_):
            r = rng.random()
            if r < 0.6:
                # (an index next to intmax would grow an array to gigabytes: keep stored indices small)
                target = rng.choice(["s", "s", "t", "t", "d", "u", "w(mod(double(k), 8) + 1)", "w(mod(k, 8) + 1)",
                                     "c{mod(double(k), 5) + 1}", "c{end + 1}", "q"])
                out.append(f"{target} = {self.expr(cls, rng.randint(1, 3))};")
            elif r < 0.8:
                out.append(f"if {self.expr(cls, 2)}, {self.stmts(cls, 1)[0]} else, {self.stmts(cls, 1)[0]} end")
            elif r < 0.9:
                out.append(f"if {self.expr(cls, 1)} {rng.choice(['>', '<', '=='])} {rng.choice(_SPECIAL_D)}, "
                           f"{rng.choice(['break', 'continue'])}; end")
            else:
                out.append(f"s = s + double({self.expr(cls, 1)});")
        return out

    def program(self) -> str:
        rng = self.rng
        cls = rng.choice(INT_CLASSES)
        init = [
            f"s = {rng.choice(['0', '1.5', cls + '(3)', 'int8(1)', 'true'])}; t = {cls}({rng.randint(-9, 9)});",
            f"d = {rng.choice(_SPECIAL_D)}; u = {rng.choice(['0', cls + '(0)'])}; q = [];",
            f"v = {rng.choice(['[4 -3 2.5 0]', cls + '([4 -3 2 0])', '[1 2 3 4 5 6 7 8]'])};",
            f"w = {rng.choice(['zeros(1, 8)', '[]', cls + '(zeros(1, 8))'])}; c = {{}};",
        ]
        body = self.stmts(cls, rng.randint(1, 4))
        return "\n".join(init + [f"for k = {self.range_(cls)}"] + body + ["end"])


def test_random_integer_loops_match(fi_path):
    rng = random.Random(20261126)
    gen = _IntLoopGen(rng)
    check_all((gen.program() for _ in range(count(600))), fi_path)


def test_random_integer_loops_are_mostly_compiled(fi_path):
    """The generated loops are not all declined: enough of them run compiled to make the comparison mean something."""
    rng = random.Random(4711)
    gen = _IntLoopGen(rng)
    compiled = total = 0
    for _ in range(400):
        code = gen.program()
        try:
            runs = _loop_runs_compiled(code, fi_path)
        except Exception:  # noqa: BLE001 - an error in the program is fine here
            continue
        total += len(runs)
        compiled += sum(1 for c, _bails in runs if c)
    # (most generated loops are declined before compiling: empty, fewer than four iterations, or an error)
    assert total >= 50 and compiled >= 0.6 * total, (compiled, total)


@pytest.mark.parametrize("cls", INT_CLASSES)
def test_random_integer_arithmetic_in_loops_matches(cls, fi_path):
    """Every operator between the loop variable and doubles, logicals and integers, near the limits of the class."""
    rng = random.Random(sum(map(ord, cls)) * 7)
    info = np.iinfo(cls)
    edges = [int(info.min), int(info.min) + 1, -1, 0, 1, 2, 3, int(info.max) - 1, int(info.max)]
    edges = [e for e in edges if info.min <= e <= info.max]
    progs = []
    for _ in range(count(40)):
        lo = rng.choice(edges)
        hi = min(lo + rng.randint(3, 9), int(info.max))
        if cls in ("int64", "uint64") and rng.random() < 0.7:
            lo, hi = rng.randint(-4, 4) if cls == "int64" else rng.randint(0, 4), 12
        other = rng.choice(_SPECIAL_D + [num(rand_double(rng, 0.5)), "true", f"{cls}({rng.choice(edges)})",
                                         f"{rng.choice(INT_CLASSES)}(2)"])
        op = rng.choice(["+", "-", "*", "/", ".*", "./", "^", "\\", "<", "==", "&"])
        lhs, rhs = ("k", other) if rng.random() < 0.6 else (other, "k")
        progs.append(f"r = []; for k = {cls}({lo}):{cls}({hi}), r(end + 1) = double({lhs} {op} {rhs}); "
                     f"x = {lhs} {op} {rhs}; end")
    check_all(progs, fi_path)
