Write a MATLAB value class `Poly` (file `Poly.m`) for polynomials with real
coefficients.

Representation:

- `p = Poly(c)` takes a numeric vector `c` of coefficients in **ascending**
  powers: `Poly([5 0 2])` is 5 + 2x². Store them in the read-only property
  `Coeffs` (`SetAccess = private`) as a row vector with trailing zeros removed
  (the highest stored coefficient is nonzero). The zero polynomial (including
  `Poly([])` and `Poly([0 0])`) is stored as `Coeffs = 0`.

Methods:

- `d = degree(p)` — the degree; the zero polynomial has degree `-1`.
- `y = at(p, x)` — evaluates p at every element of the numeric array `x`;
  `y` has the same size as `x`.
- `q = deriv(p)` — the derivative, as a `Poly`.
- `p + q`, `p - q`, `-p`, `p * q` — sum, difference, negation and product.
  In `+`, `-` and `*`, either operand may also be a numeric scalar (a constant
  polynomial).
- `p == q` — logical scalar, true when the stored coefficient vectors are
  identical.
- `[q, r] = divmod(p, d)` — polynomial long division: `p = q*d + r` with
  `degree(r) < degree(d)`. Both outputs are `Poly` objects. If `d` is the zero
  polynomial, throw an error with identifier `'Poly:divideByZero'`.

Example:

```matlab
p = Poly([-1 0 1]);            % x^2 - 1
[q, r] = divmod(p, Poly([1 1]));
q.Coeffs                       % [-1 1]   (x - 1)
r.Coeffs                       % 0
at(p * 2, [0 3])               % [-2 16]
```
