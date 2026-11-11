Measurements from several runs have different lengths and are stored in a
cell array. Write a MATLAB function

    M = pad_to_matrix(c, fill)

where `c` is a cell vector (row or column) whose elements are numeric vectors
(row, column or empty). Return a matrix with one **row** per cell element, in
order; row `k` holds the elements of `c{k}` followed by copies of `fill` up to
the length of the longest vector. `fill` is optional and defaults to `NaN`.

`M` has size `numel(c)-by-L`, where `L` is the largest vector length (so `L`
can be 0). For an empty cell array return `zeros(0, 0)`.

Example:

    pad_to_matrix({[1 2 3], [4; 5], [], 6}, 0)
    % returns [1 2 3; 4 5 0; 0 0 0; 6 0 0]
