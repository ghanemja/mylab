Write a MATLAB function `[r, nsteps] = bisect_root(f, a, b, tol)` that finds a
root of the scalar function handle `f` in the interval `[a, b]` (with `a < b`)
by bisection. Do not use `fzero`.

Follow these rules exactly so that the step count is well defined:

1. If `f(a)` and `f(b)` have the same strict sign (their product is positive),
   throw an error.
2. If `f(a) == 0`, return `r = a` and `nsteps = 0`. Otherwise, if `f(b) == 0`,
   return `r = b` and `nsteps = 0`.
3. While `b - a > 2*tol`: compute the midpoint `m = (a + b)/2` and count one
   step. If `f(m) == 0`, return `r = m` immediately. Otherwise keep the half
   interval whose endpoints have opposite signs of `f`.
4. When the loop ends, return the midpoint of the final interval as `r`.

`nsteps` is the number of midpoints evaluated. With these rules the returned
root is within `tol` of a true root.

Example: a flat plate absorbs 1500 W/m^2 and loses heat by radiation
(emissivity 0.8) and convection (h = 10 W/m^2K) to surroundings at 293 K. Its
temperature T solves
`0.8*5.67e-8*(T^4 - 293^4) + 10*(T - 293) - 1500 = 0`.
`bisect_root(g, 293, 1000, 1e-6)` returns about `380.92` (kelvin).

- `[r, n] = bisect_root(@(x) cos(x) - x, 0, 1, 1e-10)` gives `r` close to
  `0.7390851332` and `n = 33`.
- `bisect_root(@(x) x - 1, 0, 2, 1e-8)` returns `r = 1` after 1 step.
