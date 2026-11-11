Write a MATLAB function `[x, y, range] = launch_path(v0, angleDeg, dt)` that
traces a projectile launched from the origin with the explicit Euler method.

The launch speed is `v0 > 0` and the launch angle `angleDeg` is in degrees
with `0 < angleDeg <= 90` (so the projectile always rises first). Gravity is `g = 9.81` downward and there is no air resistance. The
initial velocity is `vx = v0*cosd(angleDeg)`, `vy = v0*sind(angleDeg)`.

Each time step of length `dt` does, in this order:

1. `x = x + vx*dt` and `y = y + vy*dt` (using the current velocity),
2. `vy = vy - g*dt` (`vx` stays constant).

Record the starting point and every new point while `y >= 0`. Stop as soon as
a step produces a point with `y < 0`; that point is **not** recorded.

Return column vectors `x` and `y` of the recorded points (first entry 0, 0)
and `range`, the x-coordinate where the straight segment between the last
recorded point and the first point below ground crosses `y = 0` (linear
interpolation).

Example: `[x, y, r] = launch_path(10, 90, 0.5)` records the heights
`[0; 5; 7.5475; 7.6425; 5.285; 0.475]` (all x are 0) and `r = 0`.
