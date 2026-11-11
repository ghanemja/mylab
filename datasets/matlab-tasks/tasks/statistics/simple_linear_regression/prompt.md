Write a MATLAB function `out = simple_regression(x, y)` that fits the line
`y = intercept + slope*x` by ordinary least squares and reports the usual
summary quantities, without the Statistics Toolbox (`fitlm`, `regress`).

`out` is a struct with fields:

- `slope`, `intercept`: the least-squares estimates;
- `r2`: the coefficient of determination `1 - SSE/SST`;
- `se_slope`, `se_intercept`: their standard errors, using the residual
  variance `s2 = SSE/(n - 2)`:
  `se_slope = sqrt(s2/Sxx)` and
  `se_intercept = sqrt(s2*(1/n + mean(x)^2/Sxx))`, where
  `Sxx = sum((x - mean(x)).^2)`;
- `residuals`: the column vector `y - (intercept + slope*x)`.

`x` and `y` are vectors (row or column) of equal length `n`. If `n < 3` or
the lengths differ, or all `x` are equal, throw an error.

Example: `simple_regression([1 2 3 4], [3 5 7 9])` gives `slope = 2`,
`intercept = 1`, `r2 = 1`, zero standard errors and zero residuals.
