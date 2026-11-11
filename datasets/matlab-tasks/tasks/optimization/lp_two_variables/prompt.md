A workshop makes two products. Write a MATLAB function
`[x, fval] = lp2d_max(c, A, b)` that solves the small linear program

    maximize  c(1)*x1 + c(2)*x2
    subject to A*[x1; x2] <= b,  x1 >= 0,  x2 >= 0

by enumerating the vertices of the feasible polygon (no `linprog`):

1. Treat each constraint, including `x1 >= 0` and `x2 >= 0`, as a line.
2. For every pair of non-parallel lines, compute their intersection point.
3. Keep the points that satisfy all constraints up to a tolerance of `1e-9`
   (relative to the size of `b`), and evaluate the objective at each.
4. Return the best point as the 2-by-1 column `x` and its objective `fval`.

`A` is `m`-by-2 and `b` is an `m`-element vector with `b >= 0` (so the origin
is feasible). You may assume the optimum is finite. If several vertices tie
for the optimum, return any of them.

Example: profits `c = [30 50]`, machine hours `[1 2; 3 2]*x <= [40; 60]` and a
market limit `x2 <= 15` give `x = [10; 15]` and `fval = 1050`.
