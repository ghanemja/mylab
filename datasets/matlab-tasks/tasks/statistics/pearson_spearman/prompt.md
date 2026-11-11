Write a MATLAB function `[r, rho] = correlations(x, y)` that returns the
Pearson correlation coefficient `r` and the Spearman rank correlation `rho` of
two vectors of equal length, without `corr`, `corrcoef` or `tiedrank`.

- First drop every pair `(x(i), y(i))` in which either value is `NaN`.
- Spearman's `rho` is the Pearson correlation of the ranks. Rank values from 1
  (smallest) upward; tied values all receive the **average** of the ranks they
  occupy (e.g. `[10 20 20 30]` has ranks `[1 2.5 2.5 4]`).
- If fewer than 2 pairs remain, or either variable is constant after
  dropping, the corresponding result is `NaN`.
- `x` and `y` may be rows or columns. If their lengths differ, throw an error.

Example: hours of study `[1 2 3 4 5 6]` against the scores
`[1 8 27 64 125 216]` give `rho = 1` (perfectly monotonic) and
`r = 0.9379` (approximately).
