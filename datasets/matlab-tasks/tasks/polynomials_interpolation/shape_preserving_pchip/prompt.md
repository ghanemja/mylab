Write a MATLAB function `yq = pchip_eval(x, y, xq)` that interpolates data
with a shape-preserving piecewise cubic Hermite interpolant (the same
interpolant as MATLAB's `pchip`), implemented by hand (no `pchip`, `interp1`,
`spline`, `ppval` or `mkpp`).

Let `h(k) = x(k+1) - x(k)` and `del(k) = (y(k+1) - y(k))/h(k)` for
`k = 1..n-1`. The slopes `d` at the data points are:

- **Interior points** `k = 2..n-1`: if `del(k-1)` and `del(k)` have opposite
  signs or either is zero, `d(k) = 0`. Otherwise use the weighted harmonic
  mean with `w1 = 2*h(k) + h(k-1)` and `w2 = h(k) + 2*h(k-1)`:
  `d(k) = (w1 + w2) / (w1/del(k-1) + w2/del(k))`.
- **End points** (shown for the left end; the right end is the mirror image
  using `h(n-1), h(n-2), del(n-1), del(n-2)`):
  `d(1) = ((2*h(1) + h(2))*del(1) - h(1)*del(2)) / (h(1) + h(2))`; then if
  `sign(d(1)) ~= sign(del(1))` set `d(1) = 0`; otherwise, if
  `sign(del(1)) ~= sign(del(2))` and `abs(d(1)) > abs(3*del(1))`, set
  `d(1) = 3*del(1)`.
- With exactly two points, both slopes equal `del(1)` (a straight line).

On each interval the interpolant is the cubic Hermite polynomial matching the
values and the slopes `d` at both ends.

- `x` is strictly increasing with at least 2 points (else throw an error);
  `y` has the same number of elements.
- Queries outside `[x(1), x(end)]` return `NaN`; `yq` has the size of `xq`.

Because of the slope rules the interpolant never overshoots monotone data:
for data that is flat between two points, it stays flat there.

Example: `pchip_eval([0 1 2 3], [0 0 1 1], [0.5 1.5 2.5])` returns
`[0 0.5 1]`.
