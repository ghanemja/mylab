Write a MATLAB function `I = simpson_composite(f, a, b, n)` that approximates
the integral of `f` from `a` to `b` with the composite Simpson's 1/3 rule on
`n` equal subintervals. Do not use `integral`, `quad` or `trapz`.

- `f` is a function handle that accepts a vector and returns values
  elementwise.
- `n` must be a positive even integer; otherwise throw an error.
- `a > b` is allowed and gives the negative of the integral from `b` to `a`
  (the formula handles this automatically when `h = (b - a)/n` is negative).

Example: a spring-like force `F(x) = 50*x/(1 + x^2)` newtons acts over
`x = 0..3` metres. `simpson_composite(@(x) 50*x./(1 + x.^2), 0, 3, 200)`
returns the work, about `57.565` joules.

- `simpson_composite(@(x) x.^3, 0, 2, 2)` returns exactly `4` (Simpson's rule
  integrates cubics exactly).
