Write a MATLAB function `[Q, R] = mgs_qr(A)` that computes the thin (economy)
QR factorization of an `m`-by-`n` matrix `A` with `m >= n` using the
**modified** Gram-Schmidt process, written by hand (no `qr`, `orth`, `null`).

- `Q` is `m`-by-`n` with orthonormal columns; `R` is `n`-by-`n` upper
  triangular with **positive** diagonal; `A = Q*R`.
- Modified Gram-Schmidt: for `k = 1..n`, normalize the current column `k`
  (its norm becomes `R(k,k)`), then immediately subtract its component from
  every later column `j > k` (the coefficient becomes `R(k,j)`). This is much
  more stable than classical Gram-Schmidt.
- If `m < n`, throw an error. If some column becomes numerically zero
  (`R(k,k) <= 1e-10 * norm(A(:,k))`, or `A(:,k)` is entirely zero), the
  columns are linearly dependent: throw an error.

Example: `[Q, R] = mgs_qr([3 1; 4 2])` gives `R = [5 2.2; 0 0.4]` and
`Q = [0.6 -0.8; 0.8 0.6]`.
