# Symbolic math

mylab can do exact math with symbols: derivatives, integrals, limits, series,
equation solving, differential equations, transforms and symbolic matrices.
The syntax is the one MATLAB uses (`syms`, `sym`, `diff`, `int`, `solve`, ...).
The work is done by [SymPy](https://www.sympy.org), a Python library.

## Install

SymPy is optional. Install it together with mylab:

```
pip install "mylab-lang[symbolic]"
```

Without SymPy every other part of mylab works as before. A symbolic function
then stops with a message that shows the install command.

In the web app nothing has to be installed. SymPy is downloaded the first time
a program uses a symbolic function. You see the note "Loading symbolic math…"
while that happens. It takes a few seconds, once per session.

## A first example

```matlab
syms x
f = x^2*sin(x);
df = diff(f, x)          % x^2*cos(x) + 2*x*sin(x)
F = int(f, x)            % an antiderivative
area = int(f, x, 0, pi)  % pi^2 - 4
double(area)             % 5.8696
```

Results are printed the way MATLAB prints them: `x^2 + 2*x + 1`, `exp(x)`,
`pi`, `1/3`, `x^(1/2)`. Matrices are printed row by row in brackets.

## What works

### Creating symbolic values

| Code | Result |
| --- | --- |
| `syms x y z` | three variables |
| `syms x real`, `syms x positive`, `syms n integer` | variables with an assumption |
| `syms f(t)` | a symbolic function `f` and its variable `t` |
| `syms a [1 3]`, `syms A [2 2]` | `[a1, a2, a3]`, a 2-by-2 matrix `A1_1 ... A2_2` |
| `sym('x')`, `sym('x', 'real')`, `sym('a', [1 3])` | the same, as a function call |
| `sym(1/3)`, `sym(pi)`, `sym(sqrt(2))`, `sym(0.1)` | exact values `1/3`, `pi`, `2^(1/2)`, `1/10` |
| `sym('1/3')`, `str2sym('x^2 + 1/2')` | values and expressions from text |
| `sym([1 2; 3 4])` | a symbolic matrix |
| `f(x, y) = x^2 + y` | a symbolic function; `f(1, 2)` gives `3` |

`class(x)` is `'sym'` (`'symfun'` for symbolic functions) and `isa(x, 'sym')`
is true. `symvar(expr)` lists the variables of an expression.
`assume(x > 0)`, `assume(x, 'real')`, `assumeAlso` and `assumptions` set and
show assumptions. A plain `syms x` clears them again.

### Arithmetic, indexing and display

All operators work between symbolic values and between symbolic values and
numbers: `+ - * / \ ^ .* ./ .^ '` and `.'`. A number that meets a symbolic
value becomes an exact symbolic number, so `x + 0.5` is `x + 1/2`.

Comparisons build equations and conditions: `x^2 == 4`, `x > 0`, `a ~= b`.
Use them with `solve`, `assume`, `isAlways`, `lhs` and `rhs`.

Symbolic matrices support indexing, assignment, `end`, deleting rows or
columns, `[A, B]` and `[A; B]`, `size`, `numel`, `sum`, `prod`, `diag`,
`reshape`, `repmat`, `dot`, `cross`, `norm` and `for` loops.

### Calculus

```matlab
syms x y n k
diff(x^3*y^2, y)                 % 2*x^3*y
diff(x^5, 3)                     % 60*x^2
int(1/(1 + x^2), -inf, inf)      % pi
limit(sin(x)/x, x, 0)            % 1
limit(1/x, x, 0, 'right')        % Inf
taylor(exp(x), x, 'Order', 4)    % x^3/6 + x^2/2 + x + 1
symsum(1/k^2, k, 1, inf)         % pi^2/6
symprod(k, k, 1, 5)              % 120
jacobian([x*y; x + y], [x y])    % [y, x; 1, 1]
```

Also: `gradient`, `hessian`, `divergence`, `curl` and `laplacian`.

### Algebra

`simplify`, `simplifyFraction`, `expand`, `factor`, `collect`, `combine`,
`rewrite`, `horner`, `numden`, `partfrac`, `coeffs`, `sym2poly`, `poly2sym`,
`polynomialDegree`, `lhs`, `rhs`, `children`, `isAlways`, `finverse`.

`subs` replaces variables or subexpressions:

```matlab
subs(x^2 + y, x, 3)              % y + 9
subs(x^2 + y, {x, y}, {1, 2})    % 3
subs(x^2 + y, [x y], [a b])      % a^2 + b
subs(x^2, x, [1 2 3])            % [1, 4, 9]
```

### Solving equations

```matlab
syms x y
solve(x^2 - 5*x + 6 == 0, x)                       % [2; 3]
[sx, sy] = solve([x + y == 3, x - y == 1], [x y])  % 2 and 1
S = solve([x + y == 3, x - y == 1], [x y]);        % a struct: S.x, S.y
vpasolve(cos(x) == x, x)                           % 0.739085133215...
```

`solve` accepts `'Real', true`, `'PrincipalValue', true` and
`'ReturnConditions', true`. `vpasolve` takes a start value or an interval.
`linsolve(A, b)` and `A \ b` solve symbolic linear systems.

### Differential equations

```matlab
syms y(t)
dsolve(diff(y, t) == -2*y, y(0) == 1)              % exp(-2*t)

Dy = diff(y, t);
dsolve(diff(y, t, 2) + y == 0, y(0) == 1, Dy(0) == 0)   % cos(t)
```

Systems work too: `[u, v] = dsolve(diff(u) == v, diff(v) == -u)`.
Free constants are called `C1`, `C2`, ...

### Transforms

`laplace`, `ilaplace`, `fourier`, `ifourier`, `ztrans`, `iztrans`.
`heaviside`, `dirac` and `piecewise` are available as building blocks.

### Matrices

`det`, `inv`, `rank`, `null`, `colspace`, `eig` (also `[V, D] = eig(A)`),
`transpose`, `trace`, `rref`, `linsolve`, `charpoly`, `adjoint`, `expm`,
`pinv`.

### Functions of symbolic values

The usual functions accept symbolic input: `sin`, `cos`, `tan` and their
inverses, the hyperbolic functions, `exp`, `log`, `log2`, `log10`, `sqrt`,
`abs`, `sign`, `real`, `imag`, `conj`, `floor`, `ceil`, `gamma`, `beta`,
`factorial`, `nchoosek`, `erf`, `erfc`, `besselj`, `bessely`, `lambertw`,
`zeta`, `sinint`, `cosint`, `mod`, `max`, `min` and more.

### From symbols to numbers, text and plots

| Code | Result |
| --- | --- |
| `double(expr)` | a normal number or matrix (the expression must have no variables) |
| `vpa(expr, d)`, `digits(d)` | numbers with `d` significant digits (default 32) |
| `matlabFunction(expr)` | a function handle that works on arrays |
| `matlabFunction(expr, 'Vars', [y x])` | the same, with a chosen argument order |
| `char(expr)`, `string(expr)` | the expression as text |
| `latex(expr)`, `pretty(expr)` | LaTeX code, a two-dimensional text layout |
| `eval(expr)` | evaluates the expression with the workspace variables |
| `fplot(expr)`, `ezplot(expr)` | a line plot of an expression in one variable |
| `fsurf(expr)`, `fmesh(expr)`, `fcontour(expr)` | plots of an expression in two variables |
| `fimplicit(x^2 + y^2 == 1)` | an implicit curve |

## Known differences from MATLAB

- **Equal results can look different.** SymPy and MATLAB do not always order
  terms or factor results the same way. For example mylab prints
  `x^2*cos(x) + 2*x*sin(x)` where MATLAB prints `2*x*sin(x) + x^2*cos(x)`.
  Compare results with `simplify(a - b)` or `isAlways(a == b)`, not as text.
- **Assumptions belong to the workspace.** `assume` and `syms x real` update
  the symbolic variables in the current workspace and the base workspace.
  Expressions stored elsewhere (inside cells, structs or other functions) keep
  the old assumptions. Only simple conditions on one variable are understood:
  `x > 0`, `x >= 2`, `x ~= 0`, and the sets `real`, `positive`, `integer`,
  `rational`.
- **`solve` lists the solutions it finds.** For periodic equations such as
  `sin(x) == 0` you get the principal solutions, not the whole family.
  `'ReturnConditions'` returns empty parameters and conditions that are all
  true. Inequalities are not solved.
- **Polynomial roots.** As in MATLAB, `solve` writes the roots of a polynomial
  factor of degree 3 or more as `root(...)` objects unless you pass
  `'MaxDegree', 3` (or 4). `double` and `vpa` turn them into numbers. This only
  applies to polynomials with number coefficients; with symbolic coefficients
  the explicit formulas are returned.
- **`int` leaves out special cases.** `int(x^n, x)` returns `x^(n + 1)/(n + 1)`
  without the case `n == -1`. Definite integrals are returned without
  convergence conditions.
- **`children`** returns a cell array (as in current MATLAB releases).
- **`sym` of a decimal number** finds exact fractions with a denominator up to
  10^12, multiples of pi and square roots. Other doubles are converted to the
  exact binary fraction they store.
- **`vpa`** prints trailing digits as SymPy rounds them. The last digit can
  differ from MATLAB, which computes with a few guard digits.
- **`matlabFunction`** returns a function handle only. Writing a file
  (`'File'`) and several outputs are not supported. The text of the handle
  differs from MATLAB's, the values agree.
- **`fourier` and `ztrans`** know fewer transforms than MATLAB. A transform
  that is not found is returned unevaluated, for example `fourier(f(x), x, w)`.
  `iztrans` handles rational functions.
- **`dsolve`** takes symbolic equations (`diff(y, t) == ...`). The old text
  form `dsolve('Dy = -2*y')` is not supported.
- **Symbolic arrays are two-dimensional.**
- **`save`** does not store symbolic variables. Save `char(expr)` and read it
  back with `str2sym`.
- **Not available:** units, `symunit`, `odeToVectorField`, `odeFunction`,
  `pdepe`-style helpers, `fold`, `mapSymType`, `symmatrix`, live-script
  formatting, and code generation other than `matlabFunction`, `latex`
  and `char`.

## Speed

Symbolic work is much slower than numeric work. For numbers in a loop, convert
the expression once with `matlabFunction` and call the handle.
