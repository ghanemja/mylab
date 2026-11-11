Write a MATLAB function `[xmin, fmin, nevals] = golden_min(f, a, b, tol)` that
minimizes a unimodal function `f` on `[a, b]` with golden-section search,
written by hand (no `fminbnd` or `fminsearch`).

- Keep two interior points `c = b - g*(b - a)` and `d = a + g*(b - a)` with
  `g = (sqrt(5) - 1)/2`. If `f(c) < f(d)` the minimum lies in `[a, d]`,
  otherwise in `[c, b]`. Reuse the function value of the interior point that
  survives, so each iteration costs **one** new evaluation of `f`.
- Stop when `b - a <= tol`. Return `xmin = (a + b)/2` and `fmin = f(xmin)`.
- `nevals` is the total number of calls to `f`, including the two initial
  interior points and the final evaluation at `xmin`.
- The returned `xmin` must be within `tol` of the true minimizer (for
  tolerances well above the round-off limit, roughly `sqrt(eps)` relative).

Example: a closed cylindrical can holding 500 cm^3 has surface area
`S(r) = 2*pi*r^2 + 1000/r`. `golden_min(@(r) 2*pi*r^2 + 1000./r, 1, 10, 1e-6)`
returns `r` about `4.3013` cm.
