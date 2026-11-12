"""Micro-benchmarks for the interpreter. Run: python benchmarks/bench.py"""

import time

from mylab import Interpreter

CASES = {
    "scalar loop (200k)": "s = 0; for k = 1:200000, s = s + k^2; end",
    "fill array by index (50k)": "x = zeros(1, 50000); for k = 1:50000, x(k) = k * 2; end",
    "read array by index (100k)": "v = rand(1, 1000); t = 0; for k = 1:100000, t = t + v(mod(k, 1000) + 1); end",
    "while + if (100k)": "n = 0; c = 0; while n < 100000, n = n + 1; if mod(n, 3) == 0, c = c + 1; end, end",
    "recursive calls fib(18)": (
        "function r = fib(n)\nif n < 2\nr = n;\nelse\nr = fib(n-1) + fib(n-2);\nend\nend\nf = fib(18);"
    ),
    "matrix 2D index (40k)": "A = zeros(200); for i = 1:200, for j = 1:200, A(i, j) = i + j; end, end",
    "vectorized (1e6)": "x = linspace(0, 1, 1e6); y = sum(sin(x).^2 + cos(x).^2);",
    "array expression (1e7)": "x = linspace(0, 1, 1e7); y = sin(x).^2 + cos(x).^2 + x.*x - 2*x;",
    "cell append (20k)": "c = {}; for k = 1:20000, c{end+1} = k; end",
}


def main():
    for name, code in CASES.items():
        m = Interpreter(out=lambda s: None)
        t0 = time.perf_counter()
        m.execute(code)
        print(f"{name:32s} {time.perf_counter() - t0:7.3f} s")


if __name__ == "__main__":
    main()
