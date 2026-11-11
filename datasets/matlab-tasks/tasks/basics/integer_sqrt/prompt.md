Write a MATLAB function

    r = integer_sqrt(n)

that returns the largest integer `r` with `r^2 <= n`, for a nonnegative
integer-valued double `n` with `n <= 2^53`.

The answer must be exact for every allowed `n`. Note that `floor(sqrt(n))`
alone is **not** always correct near the top of this range, because `sqrt`
rounds its result to the nearest double; your function must correct for that.
Return `r` as a double.

Examples:

- `integer_sqrt(15)` returns `3`.
- `integer_sqrt(16)` returns `4`.
- `integer_sqrt(0)` returns `0`.
