A surveyor measured ground elevations at several points. Write a MATLAB
function `[coef, rms] = fit_plane(P)` that fits the plane `z = a*x + b*y + c`
to the points in the least-squares sense.

- `P` is an `N`-by-3 matrix with rows `[x y z]`.
- `coef` is the 3-by-1 column `[a; b; c]`.
- `rms` is the root-mean-square of the vertical residuals,
  `sqrt(mean((a*x + b*y + c - z).^2))`.
- If `N < 3`, throw an error.

Example: the points `[0 0 1; 1 0 3; 0 1 4; 1 1 6]` lie on
`z = 2x + 3y + 1`, so the function returns `coef = [2; 3; 1]` and `rms = 0`
(up to round-off).
