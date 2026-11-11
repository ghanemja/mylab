Write a MATLAB function `[x, iters, converged] = fixed_point(g, x0, tol, maxit)`
that looks for a fixed point `x = g(x)` by repeated substitution
`x_new = g(x_old)`.

- `x0` may be a scalar or a vector; `g` returns a value of the same size.
- After each evaluation of `g`, stop if `max(abs(x_new - x_old)) < tol`. Then
  return `x = x_new`, `converged = true`, and `iters` = number of calls to `g`.
- If `maxit` calls are made without meeting the test, return the last iterate,
  `iters = maxit` and `converged = false` (no error).

Example: `[x, k, ok] = fixed_point(@cos, 1, 1e-12, 500)` returns
`x` close to `0.7390851332151607`, `ok = true`, and `k = 69`.
