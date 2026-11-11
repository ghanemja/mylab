Write a MATLAB function `tz = zero_crossing_times(t, x)` that returns the times
at which the sampled signal `x(t)` crosses zero.

- If two consecutive samples have strictly opposite signs
  (`x(k)*x(k+1) < 0`), report one crossing at the time where the straight line
  between `(t(k), x(k))` and `(t(k+1), x(k+1))` is zero.
- Samples that are exactly zero need care. A run of one or more consecutive
  zero samples `x(i..j)` is a crossing only if the nonzero samples just before
  (`x(i-1)`) and just after (`x(j+1)`) the run have opposite signs; the
  crossing time is then `t(i)`. A run where the signal only touches zero (same
  sign on both sides), or a run at the very start or end of the signal, is not
  a crossing.
- `tz` is a row vector of increasing times; if there are no crossings it is
  `zeros(1, 0)`.
- `t` and `x` are vectors with the same number of elements, `t` increasing.

Example: `zero_crossing_times(0:4, [1 -1 -2 0 3])` returns `[0.5 3]`.
