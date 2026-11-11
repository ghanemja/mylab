Write a MATLAB function `y = rc_lowpass(x, fs, fc)` that applies the discrete
equivalent of an RC low-pass filter with cutoff frequency `fc` (Hz) to the
signal `x` sampled at `fs` (Hz). Do not use `filter`.

With `dt = 1/fs`, `RC = 1/(2*pi*fc)` and `alpha = dt/(RC + dt)`:

- `y(1) = x(1)` (the filter starts settled at the first sample);
- `y(k) = y(k-1) + alpha*(x(k) - y(k-1))` for `k >= 2`.

`y` has the same size and orientation as `x`. Empty input gives empty output.

Example: with `fs = 1000` and `fc = 20`, `alpha` is about `0.11164`, so the
response to `[0 1 1 1]` is about `[0 0.11164 0.21082 0.29892]`.
