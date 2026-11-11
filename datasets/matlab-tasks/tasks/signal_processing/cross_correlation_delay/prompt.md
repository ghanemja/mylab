Two microphones record the same sound; the second recording `y` is a delayed,
scaled and slightly noisy copy of the first recording `x`. Write a MATLAB
function `[lag, peak] = estimate_delay(x, y, maxlag)` that estimates the delay
in samples from the raw cross-correlation, computed by hand (no `xcorr`,
`conv` or `fft`).

- For each integer lag `L` from `-maxlag` to `maxlag`, compute
  `c(L) = sum over n of x(n)*y(n + L)`, where terms with an index outside
  either vector are left out (treated as zero). `x` and `y` may have different
  lengths and may be rows or columns.
- `lag` is the `L` with the largest `c(L)`. If several lags share the largest
  value, choose the one with the smallest `abs(L)`, and if still tied the
  negative one.
- `peak = c(lag) / sqrt(sum(x.^2) * sum(y.^2))`, a normalized similarity
  (1 for a perfect positive match with no truncation).
- A positive `lag` means `y` lags behind `x`.
- If `x` or `y` is all zeros, throw an error.

Example: `estimate_delay([0 1 2 1 0 0], [0 0 0 1 2 1], 4)` returns `lag = 2`
and `peak = 1`.
