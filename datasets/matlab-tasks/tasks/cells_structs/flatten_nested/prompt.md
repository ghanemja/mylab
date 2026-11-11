Write a MATLAB function

    out = flatten_nested(c)

that flattens a cell array whose elements may themselves be cell arrays,
nested to any depth. Return a `1-by-N` cell array containing every non-cell
value found, in depth-first order (within each cell array, elements are
visited in linear index order). Non-cell values, including empty numeric
arrays and empty chars, are kept as they are; empty cell arrays contribute
nothing. For an input with no non-cell values return `cell(1, 0)`.

Example:

    flatten_nested({1, {2, {3, 'abc'}, {}}, [4 5], {6}})
    % returns {1, 2, 3, 'abc', [4 5], 6}
