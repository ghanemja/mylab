Write a MATLAB function

    d = base_digits(n, b)

that writes the nonnegative integer `n` in base `b` (an integer, `b >= 2`) and
returns its digits as a **row vector of doubles**, most significant digit first.
Each digit is a number between `0` and `b-1` (no letters). The number zero is
written as the single digit `0`.

Examples:

- `base_digits(10, 2)` returns `[1 0 1 0]`.
- `base_digits(255, 16)` returns `[15 15]`.
- `base_digits(0, 7)` returns `0`.
