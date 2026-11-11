Write a MATLAB function `Z = column_zscores(X)` that standardizes data without
the Statistics Toolbox's `zscore`:

- If `X` is a matrix with more than one row and column, each column is
  standardized separately: subtract the column mean and divide by the column's
  sample standard deviation (normalized by `N - 1`).
- If `X` is a vector (row or column), all of its elements are standardized
  together and `Z` keeps the orientation of `X`.
- A column (or vector) whose standard deviation is zero becomes all zeros.
- `Z` has the same size as `X`. An empty input returns an empty output.

Example: `column_zscores([1 10; 3 10; 5 10])` returns `[-1 0; 0 0; 1 0]`.
