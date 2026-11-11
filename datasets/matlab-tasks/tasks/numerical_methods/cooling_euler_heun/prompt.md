An object cools according to Newton's law, `dT/dt = -k*(T - Tamb(t))`, where
the ambient temperature `Tamb(t)` may vary with time.

Write a MATLAB function `T = cooling_profile(T0, Tamb, k, dt, n, method)` that
takes `n` steps of size `dt` starting from `T(0) = T0` at time 0.

- `Tamb` is a function handle of time (for a constant ambient use
  `@(t) 20`).
- `method` is `'euler'` (explicit Euler) or `'heun'` (Heun's method: an Euler
  predictor followed by the average of the slopes at both ends of the step).
  Any other value throws an error. Do not use `ode45` or similar.
- `T` is an `(n+1)`-by-1 column vector: the temperatures at times
  `0, dt, ..., n*dt`.

Example: `cooling_profile(90, @(t) 20, 0.1, 1, 2, 'euler')` returns
`[90; 83; 76.7]`.
