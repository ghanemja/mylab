Write a MATLAB function `[p, dp] = horner_eval(c, x)` that evaluates the
polynomial with coefficient vector `c` (highest power first, the same
convention as `polyval`) and its first derivative at every element of `x`,
using Horner's scheme (one pass, no powers). Do not use `polyval`,
`polyder` or `polyfit`.

- `x` may be a scalar, vector or matrix; `p` and `dp` have the size of `x`.
- An empty `c` is the zero polynomial.

Hint: inside the Horner loop, update the derivative before the value:
`dp = dp.*x + p; p = p.*x + c(k)`.

Example: for `c = [1 -3 2]` (`x^2 - 3x + 2`),
`[p, dp] = horner_eval(c, [0 1 3])` gives `p = [2 0 2]` and `dp = [-3 -1 3]`.
