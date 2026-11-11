Write a MATLAB function `[x, iters] = grad_descent(gradf, x0, alpha, tol, maxit)`
that minimizes a smooth function by gradient descent with a fixed step size
`alpha`.

- `gradf(x)` returns the gradient at the column vector `x` as a column vector.
  `x0` may be a row or column; the returned `x` is a column.
- Loop: evaluate `g = gradf(x)`. If `norm(g) < tol`, stop. Otherwise, if
  `maxit` steps have already been taken, stop; else take the step
  `x = x - alpha*g`.
- `iters` is the number of steps taken (0 if `x0` already satisfies the test,
  at most `maxit`).

Example: for `f(x) = 0.5*x'*Q*x - b'*x` with `Q = [3 1; 1 2]` and
`b = [1; -1]`, the gradient is `Q*x - b`, and
`grad_descent(@(x) [3 1; 1 2]*x - [1; -1], [0 0], 0.3, 1e-10, 1000)` returns
`x` close to `Q\b = [0.6; -0.8]`.
