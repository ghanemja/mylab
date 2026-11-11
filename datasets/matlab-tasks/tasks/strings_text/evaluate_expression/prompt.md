Write a MATLAB function

    v = evaluate_expression(s)

that evaluates an arithmetic expression given as a char vector **without**
using `eval`, `evalin`, `str2num`, `str2func` or similar functions that run
MATLAB code.

The expression may contain:

- nonnegative decimal numbers such as `12`, `3.5`, `.5` or `4.` (no
  exponents);
- the binary operators `+ - * /` with the usual precedence (`*` and `/` bind
  tighter than `+` and `-`) and left associativity (`8 - 3 - 2` is `3`);
- unary minus, which may be repeated (`--3` is `3`) and binds tighter than
  `*` and `/`;
- parentheses, and any amount of whitespace between tokens.

Division follows IEEE arithmetic (`1/0` is `Inf`). Return a double scalar. If
the expression is malformed (empty, unknown character, missing operand,
unbalanced parentheses, two numbers in a row, a number with two decimal
points, ...), throw an error with identifier `evaluate_expression:syntax`.

Examples:

- `evaluate_expression('1 + 2 * 3')` returns `7`.
- `evaluate_expression('-(2.5 * 4) + -3')` returns `-13`.
- `evaluate_expression('64 / 4 / 2')` returns `8`.
