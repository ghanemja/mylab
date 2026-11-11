Write a MATLAB function

    v = interleave_vectors(a, b)

that merges two numeric vectors by taking elements alternately from each,
starting with `a`: `a(1), b(1), a(2), b(2), ...`. When one vector runs out,
the remaining elements of the other are appended in order.

`a` and `b` may each be a row vector, a column vector or empty. The result is
always a **row vector** (`1-by-0` when both inputs are empty).

Examples:

- `interleave_vectors([1 2 3], [10 20 30])` returns `[1 10 2 20 3 30]`.
- `interleave_vectors([1 2 3 4], 9)` returns `[1 9 2 3 4]`.
- `interleave_vectors([], [5; 6])` returns `[5 6]`.
