Write a MATLAB function `[L, U] = doolittle_lu(A)` that factors a square matrix
as `A = L*U` **without pivoting**, where `L` is unit lower triangular (ones on
the diagonal) and `U` is upper triangular. Implement the factorization
yourself; do not call `lu`.

- If `A` is not square, throw an error.
- If a pivot `U(k,k)` turns out to be zero (more precisely
  `abs(U(k,k)) <= 1e-12*norm(A, inf)`) for some `k < n`, the factorization
  without pivoting does not exist: throw an error. (A zero in the last
  diagonal position is allowed; it just means `A` is singular.)

Example: `[L, U] = doolittle_lu([2 1; 6 8])` gives `L = [1 0; 3 1]` and
`U = [2 1; 0 5]`.
