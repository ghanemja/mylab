Write a MATLAB function

    y = apply_repeatedly(f, x, n)

that applies the function handle `f` to `x` exactly `n` times, feeding each
result into the next call: `y = f(f(...f(x)...))`. `n` is a nonnegative
integer; for `n = 0` return `x` unchanged. `x` can be any value `f` accepts,
and the intermediate results may change size or class. If `n` is negative or
not a whole number, throw an error with identifier `apply_repeatedly:badCount`.

Examples:

- `apply_repeatedly(@(v) v * 2, 1, 10)` returns `1024`.
- `apply_repeatedly(@(s) [s 'ab'], 'x', 3)` returns `'xababab'`.
- `apply_repeatedly(@sqrt, 81, 0)` returns `81`.
