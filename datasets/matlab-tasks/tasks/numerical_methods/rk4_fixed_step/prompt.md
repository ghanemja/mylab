Write a MATLAB function `[t, Y] = rk4_integrate(f, tspan, y0, h)` that solves
the initial value problem `y' = f(t, y)`, `y(tspan(1)) = y0` with the classical
fourth-order Runge-Kutta method and a fixed step `h`, written by hand (no
`ode45` or other solvers).

- `f(t, y)` takes a scalar `t` and a **column** vector `y` and returns a column
  vector of the same length.
- `tspan = [t0 tf]` with `tf > t0`. `y0` may be a row or column vector.
- Time grid: let `N = ceil((tf - t0)/h - 1e-10)`. The grid is
  `t0, t0 + h, ..., t0 + (N-1)*h, tf`, i.e. all steps have length `h` except
  the last, which is shortened (or exact) so the grid ends exactly at `tf`.
- Output `t` is an `(N+1)`-by-1 column; `Y` is `(N+1)`-by-`numel(y0)`, with
  row `k` holding the solution at `t(k)` (the same layout as `ode45`).

Example: a predator-prey model with prey `x` and predators `p`,
`x' = 0.8x - 0.04xp`, `p' = -0.5p + 0.01xp`, `x(0) = 60`, `p(0) = 20`,
integrated to `t = 10` with `h = 0.01`, gives about `Y(end, :) = [59.996 20.083]`
(within about 1e-5 of the exact solution).

For `tspan = [0 1]` and `h = 0.3` the grid is `[0; 0.3; 0.6; 0.9; 1]`.
