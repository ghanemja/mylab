A temperature logger marks missing readings as `NaN`. Write a MATLAB function
`s = nan_summary(x)` that summarizes the valid (non-NaN) values of the vector
`x` in a struct with these fields:

| field    | meaning                                              |
|----------|------------------------------------------------------|
| `count`  | number of non-NaN values                             |
| `mean`   | their mean                                           |
| `median` | their median                                         |
| `std`    | their sample standard deviation (divide by `N - 1`)  |
| `min`    | their minimum                                        |
| `max`    | their maximum                                        |

- With exactly one valid value, `std` is `0`.
- If there are no valid values (all NaN, or `x` empty), `count` is `0` and all
  other fields are `NaN`.

Example: `s = nan_summary([2 NaN 4 9])` gives `s.count = 3`, `s.mean = 5`,
`s.median = 4`, `s.std = 3.6056` (approximately), `s.min = 2`, `s.max = 9`.
