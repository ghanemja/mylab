"""Micro-benchmark of the fast numeric layer (mylab/fastnum.py).

Times each operation at small sizes (1, 3 and 8 elements) and at a large size (one million
elements, or a 400x400 matrix for the linear-algebra rows) with the layer switched off and on,
and prints one table row per operation and size.

    python benchmarks/fastnum_micro.py            # the whole table
    python benchmarks/fastnum_micro.py sum mod    # only operations whose name contains a word
    python benchmarks/fastnum_micro.py --markdown # as a Markdown table

Times are per call, best of several repeats; calls are made the way the interpreter makes them
(operators through ``ops.binop``, built-ins through ``Interpreter.call_builtin``).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from mylab import Interpreter, fastnum, ops  # noqa: E402
from mylab.builtins import BUILTINS  # noqa: E402

SMALL = (1, 3, 8)
LARGE = 1_000_000


def vector(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + n)
    return (rng.random((1, n)) * 100 + 0.5).copy()


def best_time(fn, budget: float = 0.12) -> float:
    """Seconds per call: best of five batches sized to fill ``budget`` seconds in total."""
    fn()
    t0 = time.perf_counter()
    fn()
    once = max(time.perf_counter() - t0, 1e-7)
    number = max(1, int(budget / 5 / once))
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def cases(interp):
    """(name, size label, callable) for every operation and size."""
    def builtin(name, *args, nargout=1):
        fn = BUILTINS[name]
        return lambda: interp.call_builtin(name, fn, list(args), nargout)

    out = []
    two = np.array([[2.0]])
    seven = np.array([[7.0]])
    for n in SMALL + (LARGE,):
        a, b = vector(n), vector(n, 1)
        label = f"{n:,}"
        for op in ("+", ".*", "./", "<", "=="):
            out.append((f"a {op} b", label, lambda op=op, a=a, b=b: ops.binop(op, a, b)))
        out.append(("a * 2", label, lambda a=a: ops.binop("*", a, two)))
        out.append(("a .^ 2", label, lambda a=a: ops.binop(".^", a, two)))
        out.append(("(a < b) & (a > 2)", label, lambda p=a < b, q=a > 2: ops.binop("&", p, q)))
        for name in ("sum", "prod", "cumsum", "max", "min", "any", "all", "abs", "sqrt", "floor", "ceil",
                     "round", "exp", "numel", "norm", "sort", "unique"):
            out.append((f"{name}(a)", label, builtin(name, a)))
        out.append(("mod(a, 7)", label, builtin("mod", a, seven)))
        out.append(("rem(a, 7)", label, builtin("rem", a, seven)))
        out.append(("dot(a, b)", label, builtin("dot", a, b)))
        out.append(("ismember(a, b)", label, builtin("ismember", a, np.floor(b))))
        if n == 1:
            out.append(("max(a, b)", label, builtin("max", a, b)))
            out.append(("size(a, 2)", label, builtin("size", a, two)))
        if n <= 8:
            pieces = [np.array([[float(k)]]) for k in range(n)]
            out.append(("[x1, x2, ...]", label, lambda pieces=pieces: ops.build_matrix([pieces])))
        else:
            out.append(("[a, b]", label, lambda a=a, b=b: ops.build_matrix([[a, b]])))
            out.append(("fft(a)", label, builtin("fft", a)))
    rng = np.random.default_rng(3)
    for n in (3, 8, 400):
        m = rng.standard_normal((n, n)) + n * np.eye(n)
        rhs = rng.standard_normal((n, 1))
        label = f"{n}x{n}"
        out.append(("A \\ b (general)", label, lambda m=m, rhs=rhs: ops.left_divide(m, rhs)))
        spd = m @ m.T
        out.append(("A \\ b (sym. pos. def.)", label, lambda spd=spd, rhs=rhs: ops.left_divide(spd, rhs)))
        tri = np.triu(m)
        out.append(("A \\ b (triangular)", label, lambda tri=tri, rhs=rhs: ops.left_divide(tri, rhs)))
        out.append(("eig(A), values", label, builtin("eig", m)))
        out.append(("eig(A), symmetric", label, builtin("eig", spd)))
        out.append(("svd(A), values", label, builtin("svd", m)))
    return out


def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:9.2f} us"
    return f"{seconds * 1e3:9.2f} ms"


def main(argv: list[str]) -> int:
    markdown = "--markdown" in argv
    words = [a for a in argv if not a.startswith("--")]
    interp = Interpreter(out=lambda s: None)
    rows = []
    fastnum.quiet_enter()  # as during a run: NumPy's floating-point warnings are off
    try:
        for name, label, fn in cases(interp):
            if words and not any(w in name for w in words):
                continue
            fastnum.set_enabled(False)
            off = best_time(fn)
            fastnum.set_enabled(True)
            on = best_time(fn)
            rows.append((name, label, off, on))
    finally:
        fastnum.set_enabled(True)
        fastnum.quiet_leave()
    if markdown:
        print("| operation | size | layer off | layer on | speed-up |")
        print("|---|---:|---:|---:|---:|")
        for name, label, off, on in rows:
            print(f"| `{name}` | {label} | {fmt(off).strip()} | {fmt(on).strip()} | {off / on:.1f}x |")
    else:
        print(f"{'operation':26s} {'size':>10s} {'layer off':>13s} {'layer on':>13s} {'speed-up':>9s}")
        for name, label, off, on in rows:
            print(f"{name:26s} {label:>10s} {fmt(off):>13s} {fmt(on):>13s} {off / on:8.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
