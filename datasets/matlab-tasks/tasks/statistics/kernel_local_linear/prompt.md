Write a MATLAB function `yq = local_linear_smooth(x, y, xq, bw)` that smooths
noisy data with **local linear regression** and a Gaussian kernel.

For each query point `q = xq(i)`:

1. Give every data point the weight `w_j = exp(-0.5*((x(j) - q)/bw)^2)`.
2. Fit a straight line `y ~ a + b*(x - q)` to all data points by weighted
   least squares with these weights.
3. The smoothed value is the fitted line evaluated at `q`, i.e. `yq(i) = a`.

Equivalently, with `W = sum(w)`, `xbar = sum(w.*x)/W`, `ybar = sum(w.*y)/W`,
`b = sum(w.*(x - xbar).*(y - ybar)) / sum(w.*(x - xbar).^2)` and
`yq(i) = ybar + b*(q - xbar)`.

- `x`, `y` are vectors of equal length (rows or columns); `bw > 0` is the
  bandwidth. `yq` has the same size as `xq`.
- If all `x` values are equal the slope is undefined: return the weighted mean
  `ybar` instead.
- Query points outside the data range are allowed (the local line
  extrapolates).
- If `x` and `y` differ in length or `bw <= 0`, throw an error.

A key property: data lying exactly on a straight line are reproduced exactly,
for any bandwidth.

Example: `local_linear_smooth([0 1 2 3], [1 3 5 7], [0.5 4], 1)` returns
`[2 9]`.
