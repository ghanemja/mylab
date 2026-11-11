Write a MATLAB function
`d = weighted_edit_distance(a, b, costIns, costDel, costSub)`.

`a` and `b` are character row vectors (either may be empty, `''`). You may turn
`a` into `b` by any sequence of these operations:

- insert one character (cost `costIns`),
- delete one character (cost `costDel`),
- replace one character with a **different** character (cost `costSub`).

Characters that already match can be kept for free. All costs are
nonnegative numbers. Return the minimum total cost `d` (a double scalar).
Comparison is case-sensitive.

Examples:

- `weighted_edit_distance('kitten', 'sitting', 1, 1, 1)` returns `3`.
- `weighted_edit_distance('ab', 'ba', 1, 1, 5)` returns `2` (delete the `a`
  at the front, insert an `a` at the end).
- `weighted_edit_distance('abc', '', 1, 2, 1)` returns `6`.
