The Darcy friction factor `lambda` of turbulent pipe flow satisfies the
Colebrook equation

    1/sqrt(lambda) = -2*log10( rr/3.7 + 2.51/(Re*sqrt(lambda)) )

where `Re` is the Reynolds number and `rr` the relative roughness
(roughness height divided by pipe diameter, `rr >= 0`).

Write a MATLAB function `lambda = colebrook_secant(Re, rr)` that solves this
equation with the **secant method**, implemented by hand (no `fzero`,
`fsolve` or `lambertw`). A convenient unknown is `s = 1/sqrt(lambda)`, which
turns the equation into `s + 2*log10(rr/3.7 + 2.51*s/Re) = 0`; starting
guesses `s = 5` and `s = 10` work well. Iterate until the change in `s` is
below `1e-14*abs(s)` (or 100 iterations).

- `Re` may be a scalar or an array; `rr` is a scalar. `lambda` has the same
  size as `Re`.
- If any `Re` is 2300 or less (the flow is not turbulent), throw an error.

The result must agree with the exact solution to a relative error of 1e-10.

Example: `colebrook_secant(1e5, 1e-4)` returns about `0.01851`.
