Write a MATLAB function `y = centered_moving_mean(x, k)` that smooths the
vector `x` with a centered moving average of odd window length `k`, written by
hand (do not use `movmean`, `smoothdata`, `filter` or `conv`).

- `y(i)` is the mean of the elements `x(i-h) ... x(i+h)` with `h = (k-1)/2`.
- Near the ends the window is truncated to the elements that exist (it is
  *not* padded).
- `NaN` values inside a window are ignored; if a window contains only `NaN`
  values, `y(i)` is `NaN`.
- `y` has the same size and orientation as `x`.
- If `k` is not a positive odd integer, throw an error.

Example: `centered_moving_mean([1 2 NaN 4 10], 3)` returns
`[1.5 1.5 3 7 7]`.
