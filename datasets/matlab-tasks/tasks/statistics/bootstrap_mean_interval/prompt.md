Write a MATLAB function `[ci, bootmeans] = bootstrap_mean_ci(x, nboot, alpha)`
that estimates a `100*(1 - alpha)`% confidence interval for the mean of the
sample `x` with the percentile bootstrap, without `bootstrp` or `bootci`.

1. Repeat `nboot` times: draw `numel(x)` indices uniformly at random **with
   replacement** (e.g. with `randi`), and record the mean of the resampled
   values.
2. `bootmeans` is the `nboot`-by-1 column of these means.
3. Sort the bootstrap means; `ci = [lo hi]` (a 1-by-2 row) where
   `lo = sorted(max(1, floor(nboot*alpha/2)))` and
   `hi = sorted(min(nboot, ceil(nboot*(1 - alpha/2))))`.

`x` is a non-empty vector, `nboot` a positive integer, `0 < alpha < 1`.
The result depends on the random number generator, so only its properties are
checked: e.g. every bootstrap mean lies between `min(x)` and `max(x)`, and for
large `nboot` the interval brackets `mean(x)`.
