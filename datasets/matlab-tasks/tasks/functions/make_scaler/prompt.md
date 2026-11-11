Write a MATLAB function

    [fwd, inv] = make_scaler(lo, hi)

that returns two function handles:

- `fwd(x)` maps values linearly so that `lo` maps to 0 and `hi` maps to 1,
  i.e. `(x - lo) / (hi - lo)`;
- `inv(y)` undoes it: `lo + y * (hi - lo)`.

Both handles must work elementwise on arrays of any shape. `lo` and `hi` are
real scalars; `hi` may be smaller than `lo`. If `lo == hi`, throw an error
with identifier `make_scaler:degenerate`. The handles must keep working with
the values of `lo` and `hi` given at creation time.

Example:

    [f, g] = make_scaler(10, 20);
    f([10 15 25])   % returns [0 0.5 1.5]
    g(0.25)         % returns 12.5
