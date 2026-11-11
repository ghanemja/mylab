Write a MATLAB function

    [y, idx] = first_success(fns, x)

`fns` is a cell array of function handles, each taking one input. Call
`fns{1}(x)`, `fns{2}(x)`, ... in order. Return the result `y` of the first call
that does **not** throw an error, and the position `idx` of that handle in
`fns`. Later handles must not be called.

If every call fails, or `fns` is empty, throw an error with identifier
`first_success:allFailed` and a message that contains the number of handles
that were tried, written in decimal (e.g. `All 3 attempts failed.`).

Example:

    fns = {@(v) v(10), @(v) v(2) * 100, @(v) v(1)};
    [y, idx] = first_success(fns, [4 5 6])   % y = 500, idx = 2
