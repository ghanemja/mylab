Write a MATLAB function `X = gauss_solve(A, B)` that solves `A*X = B` by
Gaussian elimination with **partial pivoting** followed by back substitution,
written by hand. Do not use `\`, `/`, `inv`, `pinv`, `lu`, `linsolve`, `rref`
or `mldivide`.

- `A` is a square `n`-by-`n` matrix. `B` is `n`-by-`m` (one or several
  right-hand sides); `X` is `n`-by-`m`.
- At elimination step `k`, swap row `k` with the row (at or below `k`) whose
  entry in column `k` has the largest absolute value.
- If `A` is not square, or `B` has the wrong number of rows, throw an error.
- If the chosen pivot satisfies `abs(pivot) <= n*eps*max(abs(A(:)))`, the
  matrix is (numerically) singular: throw an error.

Example: `gauss_solve([0 2 1; 3 -1 2; 1 1 4], [5; 7; 13])` needs a row swap
because the first pivot is zero, and returns `[0.875; 1.125; 2.75]`.
