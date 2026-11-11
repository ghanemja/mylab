Write a MATLAB function `c = in_exactly_one(a, b)`.

`a` and `b` are numeric arrays of any shape (possibly empty) containing no
`NaN`. Return a **row** vector of the distinct values that occur in exactly
one of the two arrays (in `a` but not in `b`, or in `b` but not in `a`),
sorted in ascending order. If there are none, return `zeros(1,0)`.

Repeated values inside one array count once.

Examples:

- `in_exactly_one([1 2 2 3], [3; 4])` returns `[1 2 4]`.
- `in_exactly_one([5 5], 5)` returns `zeros(1,0)`.
