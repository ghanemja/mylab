Write a MATLAB function `x = box_least_squares(A, b, lo, hi)` that solves

    minimize 0.5*norm(A*x - b)^2   subject to  lo <= x <= hi

(element-wise bounds) without `lsqlin`, `lsqnonneg`, `quadprog` or other
toolbox solvers.

- `A` is `m`-by-`n` with full column rank, `b` has `m` elements, `lo` and
  `hi` have `n` elements with `lo <= hi`; entries may be `-Inf`/`Inf` (no
  bound). All vectors may be rows or columns; return `x` as an `n`-by-1
  column.
- A simple method that works: projected gradient descent. Start from the
  projection of `zeros(n,1)` onto the box, use the step `1/L` with
  `L = norm(A)^2`, take `x = min(max(x - (A'*(A*x - b))/L, lo), hi)`, and stop
  when the change in `x` is tiny (e.g. `norm(dx) <= 1e-13*(1 + norm(x))`) or
  after many iterations. An accelerated variant (FISTA) converges faster.
- The answer must be within `1e-6` of the exact constrained minimizer.

Example: with `A = eye(2)`, `b = [2; -3]`, `lo = [0; 0]`, `hi = [1; 1]` the
answer is the clipped vector `[1; 0]`.
