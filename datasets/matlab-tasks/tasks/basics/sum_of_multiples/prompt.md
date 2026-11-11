Write a MATLAB function `s = sum_of_multiples(factors, limit)`.

It returns the sum of all distinct positive integers smaller than `limit` that
are divisible by at least one of the numbers in the vector `factors`. Ignore
factors that are zero. If no number qualifies, return 0.

Examples:

- `sum_of_multiples([3 5], 10)` returns `23` (3 + 5 + 6 + 9).
- `sum_of_multiples([7], 7)` returns `0`.
