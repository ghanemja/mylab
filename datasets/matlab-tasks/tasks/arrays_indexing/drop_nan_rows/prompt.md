A sensor log is stored as a numeric matrix with one reading per row. Rows with
a missing value (`NaN`) in any column must be thrown away before analysis.

Write a MATLAB function

    [B, removed] = drop_nan_rows(A)

- `B` is `A` with every row that contains at least one `NaN` removed. The
  remaining rows keep their order and `B` keeps the number of columns of `A`
  (so removing all rows of a 3-by-4 matrix gives a 0-by-4 matrix).
- `removed` is a **column vector** with the original row numbers of the
  removed rows, in increasing order (`0-by-1` when nothing is removed).

Example:

    A = [1 2; NaN 4; 5 6; 7 NaN];
    [B, removed] = drop_nan_rows(A)
    % B = [1 2; 5 6], removed = [2; 4]
