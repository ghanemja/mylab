Write a MATLAB function `idx = insert_position(v, x)`.

`v` is a numeric vector sorted in non-decreasing order (it may contain repeated
values) and `x` is a numeric scalar. Return the smallest index `idx` such that
`v(idx) >= x`. If every element of `v` is smaller than `x` (including when `v`
is empty), return `numel(v) + 1`. In other words, `idx` is the leftmost
position where `x` could be inserted while keeping `v` sorted.

Use binary search so the function needs only about log2(n) comparisons; it
must handle vectors with hundreds of thousands of elements quickly. The result
is a double scalar.

Examples:

- `insert_position([1 3 3 5], 3)` returns `2`.
- `insert_position([1 3 3 5], 4)` returns `4`.
- `insert_position([1 3 3 5], 9)` returns `5`.
