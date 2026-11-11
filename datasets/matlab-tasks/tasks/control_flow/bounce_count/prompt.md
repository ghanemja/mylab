A ball is dropped from height `h0`. Each time it hits the floor it bounces back
up to `r` times the height it fell from (`0 <= r < 1`). Write a MATLAB function

    n = bounce_count(h0, r, hmin)

that returns how many bounces reach a height of **at least** `hmin`. Use a loop
that follows the ball bounce by bounce. All inputs are nonnegative scalars,
`hmin > 0`.

Examples:

- `bounce_count(10, 0.5, 1)` returns `3` (rebounds 5, 2.5, 1.25, then 0.625).
- `bounce_count(10, 0.5, 1.25)` returns `3` (a rebound equal to `hmin` counts).
- `bounce_count(1, 0.9, 2)` returns `0`.
