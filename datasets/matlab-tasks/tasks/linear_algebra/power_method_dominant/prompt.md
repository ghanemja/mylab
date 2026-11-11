Write a MATLAB function `[lam, v, iters] = power_iteration(A, x0, tol, maxit)`
that estimates the eigenvalue of largest magnitude of the square matrix `A`
and a matching eigenvector with the power method (do not use `eig` or `eigs`).

Algorithm:

1. `v = x0(:) / norm(x0)`; `lam = v' * A * v`.
2. Repeat for `k = 1, 2, ..., maxit`: `w = A*v`, `v = w / norm(w)`, new
   `lam = v' * A * v` (Rayleigh quotient). Stop as soon as
   `abs(lam_new - lam_old) <= tol * abs(lam_new)`; `iters` is the number of
   multiplications by `A` performed.
3. If the test is never met, return the values after `maxit` steps.

Normalize the returned `v` to unit 2-norm and choose its sign so that its
entry of largest absolute value is positive (the first such entry on ties).
`v` is a column vector.

Example: for the stiffness-like matrix
`A = [4 -1 0 0; -1 4 -1 0; 0 -1 4 -1; 0 0 -1 3]` and `x0 = [1; 0; 0; 0]`,
`power_iteration(A, x0, 1e-14, 1000)` returns `lam` about `5.5321`.
