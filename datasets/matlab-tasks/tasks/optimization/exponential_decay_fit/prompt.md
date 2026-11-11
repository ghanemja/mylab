The activity of a radioactive tracer is measured at times `t`. Write a MATLAB
function `p = fit_decay(t, y)` that fits the model `y = A*exp(-k*t)` by
minimizing the sum of squared residuals `sum((A*exp(-k*t) - y).^2)` with
`fminsearch` (core MATLAB; no Curve Fitting or Optimization Toolbox).

- `t`, `y` are vectors of equal length with at least two points; `y` values
  are positive in the test data.
- Return `p = [A k]` as a 1-by-2 row.
- Use a sensible starting point: fit a straight line to `log(y)` versus `t`
  (e.g. with `polyfit`) and use `A0 = exp(intercept)`, `k0 = -slope`.
- Tighten the stopping tolerances, e.g.
  `optimset('TolX', 1e-10, 'TolFun', 1e-12, 'MaxFunEvals', 1e4, 'MaxIter', 1e4)`,
  so that the parameters are accurate to about 1e-6.

Note that this fit (least squares on `y` itself) differs from the log-linear
fit when the data are noisy.

Example: data generated exactly from `A = 12`, `k = 0.45` give back
`p = [12 0.45]` (to about 1e-6).
