An elevator stands at floor `start` and has a list of requested floors.
It serves them with a simple sweep strategy:

1. A request for the floor it is on (`start`) is served immediately.
2. Its direction is set by the **first** request in the list that differs from
   `start`: up if that floor is higher, down if lower.
3. It travels in that direction, stopping at every requested floor on the way,
   until no requests remain in that direction; then it turns around and serves
   the remaining requests in the other direction.

Each floor is served once even if it was requested several times.

Write a MATLAB function

    [order, distance] = elevator_sweep(start, requests)

that returns the floors in the order they are served (a **row vector**,
`1-by-0` if `requests` is empty) and the total number of floors travelled.
Floors are integers and may be negative (basement levels); `requests` is a row
or column vector.

Examples:

- `[o, d] = elevator_sweep(5, [7 2 9 5 3])` gives `o = [5 7 9 3 2]`, `d = 11`.
- `[o, d] = elevator_sweep(10, [4 12 8])` gives `o = [8 4 12]`, `d = 14`.
