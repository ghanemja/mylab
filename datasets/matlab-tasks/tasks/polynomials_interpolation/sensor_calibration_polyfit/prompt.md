A temperature sensor returns raw ADC counts. Reference temperatures were
recorded with a lab thermometer at the same moments. Write a MATLAB function
`[c, rmse] = calibrate_sensor(raw, ref, deg)` that fits a calibration
polynomial of degree `deg` mapping raw counts to temperature in the
least-squares sense (`polyfit` and `polyval` are fine).

- `c` is the row vector of `deg + 1` coefficients, highest power first (as
  returned by `polyfit`).
- `rmse` is the root-mean-square error of the calibrated readings against the
  reference, `sqrt(mean((polyval(c, raw) - ref).^2))`.
- `raw`, `ref` are vectors (row or column) of equal length. If there are fewer
  than `deg + 1` points, throw an error.

Example: perfectly linear data `raw = [100 200 300]`, `ref = [5 10 15]` with
`deg = 1` give `c = [0.05 0]` and `rmse = 0` (up to round-off).
