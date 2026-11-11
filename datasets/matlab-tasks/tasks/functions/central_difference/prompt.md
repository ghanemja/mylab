Write a MATLAB function

    d = central_difference(f, x, h)

that estimates the derivative of the function handle `f` at the points `x`
with the central difference formula

    d = (f(x + h) - f(x - h)) / (2*h)

`x` is a numeric array of any shape and `f` works elementwise, so `d` has the
same size as `x`. The step `h` is optional: when it is omitted, use
`h = 1e-5`.

Examples:

- `central_difference(@sin, 0)` returns a value within `1e-9` of `1`.
- `central_difference(@(t) t.^2, [1 2], 0.1)` returns `[2 4]` (up to rounding).
