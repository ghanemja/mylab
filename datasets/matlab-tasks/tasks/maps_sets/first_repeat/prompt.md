Write a MATLAB function `[val, idx] = first_repeat(v)`.

Scan the numeric vector `v` (which contains no `NaN`) from left to right and
find the first position `idx` whose value has already appeared at an earlier
position. Return that value as `val` and the position as `idx`.

If no value repeats (including for an empty vector), return `val = NaN` and
`idx = 0`.

Examples:

- `[val, idx] = first_repeat([4 7 1 7 4])` gives `val = 7`, `idx = 4`
  (the 4 repeats too, but only at position 5).
- `[val, idx] = first_repeat([1 2 3])` gives `val = NaN`, `idx = 0`.
