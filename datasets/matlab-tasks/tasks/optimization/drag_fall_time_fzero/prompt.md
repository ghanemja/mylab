An object of mass `m` (kg) is dropped from rest and falls under gravity
(`g = 9.81` m/s^2) with quadratic air drag force `c*v^2`. The distance fallen
after time `t` is

    d(t) = (m/c) * log(cosh(t*sqrt(g*c/m)))

Write a MATLAB function `t = fall_time(h, m, c)` that returns the time needed
to fall a height `h` (m) by solving `d(t) = h` numerically with `fzero` (a
bracketing interval such as `[0, tmax]` with a safe upper bound works well).

- If `c == 0` (no drag), return `sqrt(2*h/g)`.
- `h > 0`, `m > 0`, `c >= 0` are scalars.
- The result must be accurate to a relative error of 1e-9; pass suitable
  options (e.g. `optimset('TolX', 1e-12)`) because the default tolerance of
  `fzero` may be looser.

Hint: for large `t`, `log(cosh(z))` overflows; `z + log((1 + exp(-2z))/2)` is
an equivalent, stable form for `z >= 0`.

Example: a skydiver (`m = 80`, `c = 0.25`) falling 100 m needs about 4.754 s.
