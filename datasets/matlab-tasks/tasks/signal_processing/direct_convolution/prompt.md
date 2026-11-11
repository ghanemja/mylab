Write a MATLAB function `y = direct_convolution(x, h)` that computes the full
linear convolution of the vectors `x` and `h` by direct summation,
`y(n) = sum over k of x(k)*h(n - k + 1)`, without `conv`, `filter`, `fft` or
`conv2`.

- `x` and `h` may be rows or columns; `y` is always a **row** vector of
  length `numel(x) + numel(h) - 1`.
- If either input is empty, return `zeros(1, 0)`.

Example: smoothing `[1 -2 3 0.5]` with the kernel `[0.25 0.5 0.25]` gives
`[0.25 0 0 1.125 1 0.125]`.
