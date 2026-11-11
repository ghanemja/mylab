Write a MATLAB function `[cost, path] = cheapest_route(W, s, t)`.

`W` is an n-by-n matrix of nonnegative travel costs for a **directed** graph:
`W(i,j)` is the cost of going directly from node `i` to node `j`, and `Inf`
means there is no direct connection. Diagonal entries are ignored. `s` and `t`
are node indices.

Return:

- `cost` — the minimum total cost of travelling from `s` to `t` (a double
  scalar). It is `0` when `s == t` and `Inf` when `t` cannot be reached.
- `path` — a row vector of node indices of a cheapest route, starting with `s`
  and ending with `t`. When `s == t`, `path` is just `s`. When `t` is
  unreachable, `path` is empty (`zeros(1,0)`).

You may assume that, whenever a route exists, the cheapest route is unique.
Zero-cost edges may occur.

Example:

```matlab
W = [Inf 4   1   Inf
     Inf Inf Inf 1
     Inf 2   Inf 6
     Inf Inf Inf Inf];
[c, p] = cheapest_route(W, 1, 4)   % c = 4, p = [1 3 2 4]
```
