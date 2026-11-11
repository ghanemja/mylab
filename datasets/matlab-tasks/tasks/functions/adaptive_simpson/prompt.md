Write a MATLAB function

    q = adaptive_simpson(f, a, b, tol)

that approximates the integral of the function handle `f` from `a` to `b`
with **adaptive Simpson quadrature**. Do not call `integral`, `quad`,
`quadgk`, `trapz` or similar built-in integrators.

Method: Simpson's rule on `[a, b]` is
`S = (b - a)/6 * (f(a) + 4 f(m) + f(b))` with `m = (a + b)/2`. Compute
Simpson's rule on the two halves as well, `S2 = S_left + S_right`. If
`|S2 - S| <= 15 * tol`, accept `S2 + (S2 - S)/15`; otherwise apply the same
procedure recursively to each half with tolerance `tol/2` and add the
results. Limit the recursion depth to 50 levels (at the limit, accept `S2`).
Reuse function values where you can.

Details:

- `f` accepts a real scalar and returns a real scalar.
- If `a > b`, the result is minus the integral from `b` to `a`; if `a == b`
  the result is `0`.
- The result must be within about `tol` of the exact integral for smooth
  functions and for mild endpoint singularities such as `sqrt(x)` at 0.

Examples:

- `adaptive_simpson(@(x) x.^2, 0, 3, 1e-8)` returns `9` (up to rounding).
- `adaptive_simpson(@exp, 0, 1, 1e-10)` is within `1e-9` of `exp(1) - 1`.
