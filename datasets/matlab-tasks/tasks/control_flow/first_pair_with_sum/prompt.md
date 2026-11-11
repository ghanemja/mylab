Write a MATLAB function

    pair = first_pair_with_sum(x, target)

that searches the numeric vector `x` for two elements at **different**
positions `i < j` with `x(i) + x(j) == target` (exact comparison). Among all
such pairs return the one with the smallest `j`; if several have that `j`,
the one with the smallest `i`. Return `pair = [i j]` as a 1-by-2 row vector,
or `[]` if there is no such pair. Stop searching as soon as the answer is
known.

Examples:

- `first_pair_with_sum([4 1 3 2 5], 5)` returns `[1 2]`.
- `first_pair_with_sum([1 6 3 3 2], 6)` returns `[3 4]`.
- `first_pair_with_sum([2 2], 3)` returns `[]`.
