Write a MATLAB function `yq = lagrange_interp(x, y, xq)` that evaluates the
unique polynomial of degree at most `n - 1` passing through the `n` points
`(x(i), y(i))` at the query points `xq`, using the Lagrange form

    P(q) = sum over i of y(i) * prod over j ~= i of (q - x(j)) / (x(i) - x(j))

directly (do not use `polyfit`, `polyval`, `interp1` or `\`).

- `x`, `y` are vectors with the same number of elements `n >= 1`; the `x`
  values must be distinct, otherwise throw an error (also throw if the lengths
  differ).
- `xq` may have any shape; `yq` has the same size. Queries outside the data
  range are extrapolated by the polynomial.

Example: through `(0, 1)`, `(1, 3)`, `(2, 7)` passes `q^2 + q + 1`, so
`lagrange_interp([0 1 2], [1 3 7], [3 -1])` returns `[13 1]`.
