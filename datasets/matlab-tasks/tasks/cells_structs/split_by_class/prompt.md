Write a MATLAB function

    [nums, texts, others] = split_by_class(c)

that sorts the elements of the cell array `c` (any shape, visited in linear
index order) into three groups:

- `nums`: a **row vector** of doubles containing every element that is a
  numeric scalar of any numeric class (`double`, `single`, `int8`, ...),
  converted to double. Logical values are **not** numeric here.
- `texts`: a `1-by-K` cell array with every element that is a char array
  (including `''`).
- `others`: a `1-by-M` cell array with all remaining elements (non-scalar
  numeric arrays, logicals, cells, structs, ...).

Each group keeps the original order. Empty groups are `zeros(1, 0)`,
`cell(1, 0)` and `cell(1, 0)`.

Example:

    [n, t, o] = split_by_class({3, 'abc', [1 2], int8(4), true})
    % n = [3 4], t = {'abc'}, o = {[1 2], true}
