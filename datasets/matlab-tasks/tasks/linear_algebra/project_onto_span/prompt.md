Write a MATLAB function `[p, r] = project_onto_span(V, x)` that computes the
orthogonal projection `p` of the vector `x` onto the subspace spanned by the
columns of the matrix `V`, and the residual `r = x - p`.

- `V` is `n`-by-`k`. Its columns may be linearly dependent (for example one
  column may be the sum of others, or a column may be zero), so you cannot
  simply invert `V'*V`. An orthonormal basis of the column space (e.g. from
  `orth`, or from a rank-revealing factorization) handles this.
- `x` is a vector with `n` elements (row or column); `p` and `r` are `n`-by-1
  columns.
- If `V` has no nonzero columns, `p` is all zeros and `r` equals `x(:)`.
- If `size(V, 1)` differs from `numel(x)`, throw an error.

Example: `project_onto_span([1; 1; 0], [2 0 5])` returns `p = [1; 1; 0]` and
`r = [1; -1; 5]`.
