Write a MATLAB function `dy = fd_derivative(y, h)` that estimates the
derivative of uniformly sampled data `y` (sample spacing `h`) with
**second-order accurate** finite differences at every sample:

- interior samples: the central difference `(y(k+1) - y(k-1)) / (2h)`;
- first sample: the one-sided formula `(-3*y(1) + 4*y(2) - y(3)) / (2h)`;
- last sample: the mirror formula `(3*y(n) - 4*y(n-1) + y(n-2)) / (2h)`.

`dy` has the same size and orientation as `y`. If `y` has fewer than 3
elements, throw an error. Do not use `gradient` (its end-point formula is only
first-order).

Example: positions of a falling ball sampled every 0.25 s,
`y = 4.9*t.^2 + 3*t + 1` with `t = 0:0.25:2`, give
`fd_derivative(y, 0.25)` equal to the exact velocity `9.8*t + 3` (the formulas
are exact for quadratics).
