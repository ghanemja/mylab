Write a MATLAB function

    varargout = describe_vector(x)

that summarises the numeric vector `x`, ignoring `NaN` values. The summary
consists of the minimum, the maximum, the mean and the count of non-`NaN`
values. How it is returned depends on how many outputs the caller asks for:

- With zero or one output, return a scalar struct with fields `min`, `max`,
  `mean` and `count`, in that order.
- With two to four outputs, return the values separately in the order
  `[mn, mx, avg, count]` (the caller may ask for just the first two or three).

If no non-`NaN` values remain (including empty `x`), `min`, `max` and `mean`
are `NaN` and `count` is `0`.

Examples:

    s = describe_vector([3 NaN 1])
    % s.min = 1, s.max = 3, s.mean = 2, s.count = 2

    [lo, hi] = describe_vector([4 -2 7])
    % lo = -2, hi = 7
