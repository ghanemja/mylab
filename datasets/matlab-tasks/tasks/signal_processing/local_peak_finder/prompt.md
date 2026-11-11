Write a MATLAB function `[pks, locs] = find_local_peaks(x, minHeight, minDist)`
that finds peaks in the vector `x` (no NaN values), without `findpeaks` or
`islocalmax`.

1. **Candidates.** A peak is a maximal run of equal values `x(i) = ... = x(j)`
   (with `i <= j`) that is strictly higher than both neighbours of the run:
   `x(i-1) < x(i)` and `x(j+1) < x(j)`. The first and last samples of `x` can
   never be peaks (they lack a neighbour). The location of a peak is the
   **first** index `i` of its run.
2. **Height.** Discard candidates with `x(i) < minHeight`.
3. **Spacing.** Visit the remaining candidates from highest to lowest (equal
   heights: smaller index first). Keep a candidate only if its location is at
   least `minDist` samples away from every candidate already kept
   (`abs(loc - kept) >= minDist`).
4. Return the kept peaks sorted by location: `locs` (indices, double) and
   `pks = x(locs)`, both **row** vectors (`zeros(1, 0)` if there are none).

Use `minHeight = -Inf` and `minDist = 1` to keep every candidate.

Example: `x = [1 3 3 3 2 5 5 6 6 4 4 7 2 2 8]` has peaks at `locs = [2 8 12]`
with `pks = [3 6 7]` (the plateau 5 5 is not a peak because the values then
rise to 6; the final 8 is an end point). With `minDist = 5` only
`locs = [2 12]` remain: 12 is kept first, 8 is too close to it.
