Write a MATLAB function `r = real_roots_in(c, a, b)` that returns the real
roots of the polynomial with coefficients `c` (highest power first) that lie
in the closed interval `[a, b]`. You may use `roots`.

- Treat a computed root as real when `abs(imag(z)) <= 1e-7*max(1, abs(z))`,
  and use its real part.
- Keep roots with `a - 1e-9 <= root <= b + 1e-9`.
- Repeated roots are reported once: after sorting, go through the roots from
  the smallest; each group consists of a root and all following roots that are
  within `1e-5` of that first root, and is reported as the mean of the group. Multiple roots computed by
  `roots` are only accurate to about `eps^(1/m)` for multiplicity `m`, so the
  merging tolerance is deliberately loose.
- Return a **row** vector sorted ascending; `zeros(1, 0)` if there are none.
- Leading zeros in `c` are ignored. A constant polynomial (including the
  zero polynomial) has no roots to report: return `zeros(1, 0)`.

Example: `real_roots_in([1 0 -1 0], -0.5, 2)` (that is `x^3 - x`) returns
`[0 1]`.
