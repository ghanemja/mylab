Steady heat conduction in a rod, discretized with finite differences, gives a
tridiagonal linear system. Write a MATLAB function `x = thomas_solve(a, b, c, d)`
that solves such a system with the Thomas algorithm (forward elimination and
back substitution in O(n) operations), without building the full matrix and
without `\`.

- `b` (length `n`) is the main diagonal, `a` (length `n-1`) the sub-diagonal
  (`a(i)` is the entry in row `i+1`, column `i`), `c` (length `n-1`) the
  super-diagonal (`c(i)` is in row `i`, column `i+1`), and `d` (length `n`)
  the right-hand side.
- Inputs may be rows or columns; `x` is an `n`-by-1 column.
- If the lengths are inconsistent, throw an error.
- You may assume no zero pivot occurs (e.g. the matrix is diagonally
  dominant).

Example: `thomas_solve([1 1], [4 4 4], [1 1], [5 6 5])` returns `[1; 1; 1]`.
