A vehicle logger records its speed `v` (m/s) at irregular times `t` (s).
Write a MATLAB function `d = cumulative_distance(t, v)` that returns the
distance travelled from the first sample up to each sample, using the
trapezoidal rule on the (possibly non-uniform) intervals. Do not use `trapz`
or `cumtrapz`.

- `t` and `v` are vectors with the same number of elements; `t` is increasing.
  If the element counts differ, throw an error.
- `d` has the same size and orientation as `v`, and `d(1) = 0`.
- A single sample gives `d = 0`.

Example: `cumulative_distance([0 1 3], [2 4 4])` returns `[0 3 11]`.
