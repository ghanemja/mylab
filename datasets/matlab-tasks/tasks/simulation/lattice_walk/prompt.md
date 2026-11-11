Write a MATLAB function `P = lattice_walk(n)` that simulates a random walk on
the integer grid.

The walker starts at the origin `(0, 0)`. In each of `n` steps it moves one
unit up, down, left or right, each direction chosen with equal probability
(use MATLAB's random number generator, e.g. `randi`).

Return an `(n+1)`-by-2 double matrix `P` of positions: row 1 is `[0 0]` and
row k+1 is `[x y]` after step k. For `n = 0`, return `[0 0]`.

Example: one possible result of `lattice_walk(3)` is

```matlab
[0 0; 1 0; 1 1; 1 0]
```
