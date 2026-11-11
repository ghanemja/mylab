Write a MATLAB function

    z = factorial_trailing_zeros(n)

that returns how many zeros the decimal representation of `n!` ends with, for a
nonnegative integer `n` (given as a double). `n` can be as large as `1e9`, so
computing `n!` itself is not an option.

Examples:

- `factorial_trailing_zeros(4)` returns `0` (4! = 24).
- `factorial_trailing_zeros(10)` returns `2` (10! = 3628800).
- `factorial_trailing_zeros(0)` returns `0`.
