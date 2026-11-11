A cam profile is specified at a few angles and must be smooth. Write a MATLAB
function `yq = natural_spline(x, y, xq)` that evaluates the **natural** cubic
spline through the points `(x(i), y(i))` at the query points `xq`, building it
yourself (no `spline`, `interp1`, `pchip`, `ppval`, `mkpp` or `csape`).

- The spline is a cubic polynomial on each interval `[x(k), x(k+1)]`, it
  passes through every data point, and its first and second derivatives are
  continuous at the interior points. "Natural" means the second derivative is
  zero at `x(1)` and `x(end)`. (Setting up the equations for the second
  derivatives at the knots gives a tridiagonal system.)
- `x` is strictly increasing with at least 3 points (otherwise throw an
  error); `y` has the same number of elements.
- Queries outside `[x(1), x(end)]` return `NaN`. `yq` has the size of `xq`.

Example: for `x = [0 1 2]`, `y = [0 1 0]` the spline is
`1.5q - 0.5q^3` on `[0, 1]` (mirrored on `[1, 2]`), so
`natural_spline([0 1 2], [0 1 0], [0.5 1.5])` returns `[0.6875 0.6875]`.
