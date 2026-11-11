Write a MATLAB function

    [start, len] = longest_rising_run(x)

that finds the longest contiguous stretch of the vector `x` in which every
element is **strictly** greater than the one before it. Return the index
`start` of its first element and its length `len`. A single element counts as
a stretch of length 1. If several stretches have the same maximum length,
return the first one. For an empty `x` return `start = 0` and `len = 0`.

`x` may be a row or column vector.

Examples:

- `[s, n] = longest_rising_run([3 1 2 5 4 6 7 8 0])` gives `s = 5`, `n = 4`.
- `[s, n] = longest_rising_run([1 1 1])` gives `s = 1`, `n = 1`.
