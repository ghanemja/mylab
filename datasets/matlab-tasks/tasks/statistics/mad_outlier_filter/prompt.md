A pressure sensor occasionally produces spikes. Write a MATLAB function
`[mask, cleaned] = mad_outliers(x, thresh)` that flags outliers with the
scaled median absolute deviation (MAD), without `isoutlier` or `rmoutliers`.

- `med = median(x)`; `smad = 1.4826 * median(abs(x - med))`.
- An element is an outlier when `abs(x - med) > thresh * smad`.
- Special case: if `smad == 0`, every element that differs from `med` is an
  outlier.
- `mask` is a logical array the same size as `x` (`true` = outlier);
  `cleaned = x(~mask)` keeps the orientation of a row or column vector.
- `x` is a non-empty vector without NaN values.

Example: `[m, c] = mad_outliers([10 11 10 12 50 11], 3)` gives
`m = [false false false false true false]` and `c = [10 11 10 12 11]`.
