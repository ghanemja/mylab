Write a MATLAB function `[x, w] = gauss_legendre(n)` that returns the nodes
`x` and weights `w` of the `n`-point Gauss-Legendre quadrature rule on
`[-1, 1]`, so that `sum(w .* g(x))` approximates the integral of `g` over
`[-1, 1]` and is exact for every polynomial of degree up to `2n - 1`.

- `x` and `w` are `n`-by-1 column vectors with `x` sorted ascending.
- Nodes and weights must be accurate to about 1e-12 for `n` up to at least 20.
- If `n` is not a positive integer, throw an error.
- Compute them yourself (for example, from the eigen-decomposition of the
  symmetric tridiagonal Jacobi matrix of the Legendre polynomials, or by
  Newton's method on the Legendre polynomial); no toolbox functions.

Examples:

- `gauss_legendre(1)` gives `x = 0`, `w = 2`.
- `gauss_legendre(2)` gives `x = [-1; 1]/sqrt(3)`, `w = [1; 1]`.
