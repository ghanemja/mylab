Write a MATLAB function `[x, iters] = newton_minimize(gradf, hessf, x0, tol, maxit)`
that minimizes a smooth function of several variables with the pure Newton
method (no line search).

- `gradf(x)` returns the gradient (column vector) and `hessf(x)` the Hessian
  matrix at the column vector `x`. `x0` may be a row or column; `x` is
  returned as a column.
- Loop: `g = gradf(x)`; if `norm(g) < tol` stop. If `maxit` steps have been
  taken, throw an error. Otherwise solve `H*p = -g` with `H = hessf(x)` and
  set `x = x + p`.
- `iters` is the number of Newton steps taken.

Example: a quadratic `f(x) = 0.5*x'*Q*x - b'*x` is minimized in exactly one
step: `newton_minimize(@(x) Q*x - b, @(x) Q, [0; 0], 1e-10, 20)` returns
`Q\b` with `iters = 1`.
