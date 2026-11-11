A traffic light cycles through three states: `'green'` for 25 seconds, then
`'yellow'` for 5 seconds, then `'red'` for 30 seconds, then back to green.

Write a MATLAB function

    [state, remaining] = traffic_light_after(state, t)

The input `state` (a lowercase char vector) is the state that has **just
started**, and `t >= 0` is the number of seconds that pass after that moment.
Return the state the light is in after `t` seconds and how many seconds
`remaining` it will still stay in that state. A state ends exactly when its
time is used up, so after exactly 25 seconds of green the light is yellow with
5 seconds remaining. `t` can be as large as `1e9`, so do not simulate every
cycle one by one. An unknown state throws an error with identifier
`traffic_light_after:badState`.

Examples:

- `[s, r] = traffic_light_after('green', 0)` gives `s = 'green'`, `r = 25`.
- `[s, r] = traffic_light_after('green', 25)` gives `s = 'yellow'`, `r = 5`.
- `[s, r] = traffic_light_after('red', 100)` gives `s = 'green'`, `r = 15`.
