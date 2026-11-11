For an `m-by-n` matrix `A`, the `k`-th anti-diagonal consists of the elements
`A(i,j)` with `i + j - 1 == k`, for `k = 1, ..., m+n-1` (it runs from the
top-right towards the bottom-left). Write a MATLAB function

    s = antidiagonal_sums(A)

that returns a `1-by-(m+n-1)` **row vector** with the sum of each
anti-diagonal. For an empty matrix return `zeros(1, 0)`.

Examples:

- `antidiagonal_sums([1 2; 3 4])` returns `[1 5 4]`.
- `antidiagonal_sums([1 2 3; 4 5 6])` returns `[1 6 8 6]`.
- `antidiagonal_sums([1; 2; 3])` returns `[1 2 3]`.
