Write a MATLAB function `[x, fx, iters] = nelder_mead(f, x0, step, tol, maxit)`
that minimizes `f` (a function of an `n`-vector) with the Nelder-Mead simplex
method, implemented by hand (no `fminsearch`).

- Initial simplex: `x0` and the `n` points `x0 + step*e_i`, where `e_i` is the
  `i`-th unit vector.
- Each iteration, order the vertices by function value (best first). Let
  `xb` be the best, `xw` the worst, `xs` the second worst, and `xc` the
  centroid of all vertices except the worst. Then use the standard
  coefficients (reflection 1, expansion 2, contraction 0.5, shrink 0.5):
  - reflection `xr = xc + (xc - xw)`;
  - if `f(xr) < f(xb)`: try expansion `xe = xc + 2*(xc - xw)` and take the
    better of `xe` and `xr`;
  - else if `f(xr) < f(xs)`: accept `xr`;
  - else contract: if `f(xr) < f(xw)`, outside contraction
    `xo = xc + 0.5*(xr - xc)`, accepted if `f(xo) <= f(xr)`; otherwise inside
    contraction `xi = xc + 0.5*(xw - xc)`, accepted if `f(xi) < f(xw)`;
  - if the contraction is rejected, shrink every vertex toward `xb`:
    `v = xb + 0.5*(v - xb)`.
- Stop when both the spread of function values `max(f) - min(f)` and the
  largest distance `max(abs(v - xb))` over all vertices and coordinates are at
  most `tol`, or after `maxit` iterations.
- Return the best vertex `x` (same orientation as `x0`), its value `fx`, and
  the number of iterations `iters`.

Example: `nelder_mead(@(p) (p(1) - 3)^2 + 2*(p(2) + 1)^2, [0 0], 0.5, 1e-10, 2000)`
returns `x` close to `[3 -1]` and `fx` close to 0.
