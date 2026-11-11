A sensor samples at 100 Hz but a dashboard only needs 10 Hz. Write a MATLAB
function `y = block_average(x, M)` that reduces the sample rate of the vector
`x` by the integer factor `M`: split `x` into consecutive non-overlapping
blocks of `M` samples and replace each block by its mean. Samples left over at
the end (fewer than `M`) are dropped.

- `y` has the orientation of `x` (row in, row out; column in, column out) and
  `floor(numel(x)/M)` elements.
- If `M` is not a positive integer, throw an error.
- If `x` has fewer than `M` samples, return an empty vector.

Example: `block_average([1 3 5 7 9 11 13], 3)` returns `[3 9]`.
