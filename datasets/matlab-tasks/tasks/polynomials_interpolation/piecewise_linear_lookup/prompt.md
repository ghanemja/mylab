An engine map is stored as a table of breakpoints `x` (strictly increasing)
and values `y`. Write a MATLAB function `yq = pw_linear(x, y, xq)` that
interpolates the table linearly at the query points `xq`, without `interp1`,
`griddedInterpolant` or `polyfit`.

- Between two neighbouring breakpoints the result lies on the straight line
  joining them; at a breakpoint it equals the table value.
- Queries outside `[x(1), x(end)]` return `NaN` (no extrapolation), as do
  `NaN` queries.
- `yq` has the same size as `xq`. `x` and `y` are vectors with the same
  number of elements (at least 2).

Example: `pw_linear([0 10 20], [0 50 40], [5 15 25])` returns `[25 45 NaN]`.
