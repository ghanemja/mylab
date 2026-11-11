Write a MATLAB function `[counts, centers] = bin_counts(x, edges)` that counts
how many values of `x` fall in each bin defined by the vector `edges`, by hand
(do not use `histcounts`, `histc`, `hist` or `discretize`).

- `edges` must be strictly increasing with at least 2 elements; otherwise
  throw an error. Bin `k` is the half-open interval
  `[edges(k), edges(k+1))`, except the **last** bin, which also includes its
  right edge: `[edges(end-1), edges(end)]`.
- Values outside `[edges(1), edges(end)]` and `NaN` values are not counted.
- `counts` is a row vector of length `numel(edges) - 1` (class double).
- `centers` is a row vector with the midpoints of the bins.
- `x` may be any shape; empty `x` gives all-zero counts.

Example: wind speeds `[2.3 4.0 5.5 7.9 8.0 10.0 12.0]` with
`edges = 0:2:10` give `counts = [0 1 2 1 2]` (8.0 goes to `[8, 10]`, 10.0 is
on the closed right edge, 12.0 is ignored) and `centers = [1 3 5 7 9]`.
