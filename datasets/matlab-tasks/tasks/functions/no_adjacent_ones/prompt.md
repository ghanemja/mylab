Write a MATLAB function

    c = no_adjacent_ones(n)

that lists every char vector of length `n` made of the characters `'0'` and
`'1'` in which no two `'1'` characters are next to each other. Return them as
an `m-by-1` **cell array** of char row vectors sorted in increasing
(dictionary) order. `n` is a nonnegative integer; for `n = 0` return `{''}`
(one empty string). A recursive solution is natural: a valid string either
starts with `'0'` followed by a valid string of length `n-1`, or with `'10'`
followed by a valid string of length `n-2`.

Examples:

- `no_adjacent_ones(1)` returns `{'0'; '1'}`.
- `no_adjacent_ones(3)` returns `{'000'; '001'; '010'; '100'; '101'}`.
