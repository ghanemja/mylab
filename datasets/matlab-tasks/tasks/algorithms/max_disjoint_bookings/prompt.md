Write a MATLAB function `k = max_disjoint_bookings(slots)`.

A single meeting room receives booking requests. `slots` is an m-by-2 matrix;
row `[s e]` is a request for the half-open time interval from `s` up to (but
not including) `e`, with `s < e`. Two requests clash when their intervals
overlap; a request ending at time 10 and another starting at time 10 do not
clash.

Return the largest number of requests that can be accepted without any clash
(a double scalar). An empty `zeros(0,2)` input returns 0.

Examples:

- `max_disjoint_bookings([1 3; 2 4; 3 5])` returns `2`.
- `max_disjoint_bookings([1 10; 2 3; 4 5; 6 7])` returns `3`.
