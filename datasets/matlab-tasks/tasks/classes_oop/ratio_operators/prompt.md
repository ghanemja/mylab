Write a MATLAB value class `Ratio` (file `Ratio.m`) for exact fractions of
integers.

Construction and storage:

- `q = Ratio(n, d)` represents n/d, where `n` and `d` are integer-valued
  doubles; `Ratio(n)` means n/1. If `d` is 0, throw an error with identifier
  `'Ratio:zeroDenominator'`.
- The value is always stored in lowest terms with a positive denominator,
  in the read-only properties (`SetAccess = private`) `Num` and `Den`.
  Zero is stored as 0/1. Example: `Ratio(6, -4)` has `Num = -3`, `Den = 2`.

Operators (each returns a new `Ratio` in lowest terms, except `==` and `<`
which return logical scalars). Either operand may also be an integer-valued
double scalar, which is treated as that integer over 1:

- `a + b`, `a - b`, `a * b`, `a / b` (division by a zero value throws
  `'Ratio:zeroDenominator'`), unary `-a`;
- `a == b` and `a < b` compare the exact values.

Conversion: `double(q)` returns `Num/Den` as a double.

Example:

```matlab
a = Ratio(1, 6) + Ratio(1, 3);   % Num = 1, Den = 2
b = 2 - a;                       % Num = 3, Den = 2
double(b / Ratio(3, 4))          % 2
```
