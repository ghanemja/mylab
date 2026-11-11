Write a MATLAB function

    [n, d] = reduce_fraction(num, den)

that reduces the fraction `num/den` (two integer-valued double scalars) to
lowest terms.

- The returned denominator `d` is always positive; the sign of the fraction is
  carried by `n`.
- A zero numerator gives `n = 0`, `d = 1`.
- If `den` is zero, throw an error with identifier
  `reduce_fraction:zeroDenominator`.

Examples:

- `[n, d] = reduce_fraction(6, 8)` gives `n = 3`, `d = 4`.
- `[n, d] = reduce_fraction(6, -8)` gives `n = -3`, `d = 4`.
- `[n, d] = reduce_fraction(0, -5)` gives `n = 0`, `d = 1`.
