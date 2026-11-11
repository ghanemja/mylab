Write a MATLAB function `[coef, yq] = newton_interp(x, y, xq)` that builds the
interpolating polynomial through the points `(x(i), y(i))` in **Newton form**
and evaluates it at `xq`, without `polyfit`, `polyval` or `interp1`.

- `coef` is a row vector of the divided differences
  `[f[x1], f[x1,x2], f[x1,x2,x3], ..., f[x1,...,xn]]`, so that

      P(q) = coef(1) + coef(2)*(q - x(1)) + coef(3)*(q - x(1))*(q - x(2)) + ...

- `yq = P(xq)` has the same size as `xq`; evaluate it with nested
  multiplication (Horner-like, starting from the last coefficient).
- The nodes need not be sorted, but they must be distinct; otherwise throw an
  error.

Example: `[c, v] = newton_interp([0 1 3], [1 2 10], 2)` gives
`c = [1 1 1]` (since `P(q) = 1 + q + q*(q - 1)`) and `v = 5`.
