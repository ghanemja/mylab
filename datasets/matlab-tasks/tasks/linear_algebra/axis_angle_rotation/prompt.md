Write a MATLAB function `R = axis_angle_rotation(u, theta)` that returns the
3-by-3 matrix of a rotation by angle `theta` (radians) about the axis through
the origin with direction `u`, following the right-hand rule (a positive angle
turns counter-clockwise when looking from the tip of `u` toward the origin).

- `u` is any nonzero 3-element vector (row or column); normalize it first.
- If `u` is the zero vector, throw an error.
- `R*p` rotates a column vector `p`.

Example: `axis_angle_rotation([0 0 1], pi/2) * [1; 0; 0]` gives `[0; 1; 0]`.
