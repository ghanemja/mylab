Write a MATLAB function `[x, hist] = newton_solve(f, df, x0, tol, maxit)` that
solves `f(x) = 0` with Newton's method, written by hand (no `fzero`/`fsolve`).
`f` and `df` are function handles for the function and its derivative, `x0` is
the starting guess.

- The update is `x_new = x_old - f(x_old)/df(x_old)`.
- Stop after the first update for which `abs(x_new - x_old) <= tol`, and return
  that `x_new` as `x`.
- `hist` is a **column** vector of every iterate, starting with `x0` and ending
  with the returned `x`, so `numel(hist)` is the number of updates plus one.
- If `df(x_old)` is exactly zero, throw an error.
- If `maxit` updates are performed without meeting the tolerance, throw an
  error.

Example: `[x, h] = newton_solve(@(x) x^2 - 2, @(x) 2*x, 1, 1e-12, 50)` returns
`x` close to `sqrt(2)` and `h(1:3)` equal to `[1; 1.5; 1.416666...]`.
