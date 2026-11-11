A temperature logger writes `NaN` when a reading fails. Write a MATLAB function

    y = fill_forward(x)

that replaces every `NaN` in the vector `x` with the most recent non-`NaN`
value before it. `NaN` values at the start of `x`, which have no earlier value,
stay `NaN`. The output has the same size and orientation as `x` (row, column
or empty).

Examples:

- `fill_forward([NaN 1 NaN NaN 4 NaN])` returns `[NaN 1 1 1 4 4]`.
- `fill_forward([2; NaN; 3])` returns `[2; 2; 3]`.
