Write a MATLAB function `G = rule_automaton(rule, init, steps)` that runs a
one-dimensional, two-state cellular automaton.

- `init` is a 1-by-n logical (or 0/1 numeric) row: the initial cells.
- `rule` is an integer from 0 to 255 that encodes the update table.
- `steps` is a nonnegative integer: the number of generations to compute.

In each generation every cell is updated simultaneously from its own state
`c` and the states `l` and `r` of its left and right neighbours in the previous
generation. Form the number `p = 4*l + 2*c + r` (0 to 7); the new state of the
cell is bit `p` of `rule`, where bit 0 is the least significant bit. Cells
beyond either end of the row are permanently 0.

Return a `(steps+1)`-by-n **logical** matrix whose first row is `init` and
whose row k+1 is generation k.

Example (rule 90 makes each cell the exclusive-or of its neighbours):

```matlab
G = rule_automaton(90, logical([0 0 0 1 0 0 0]), 3)
% rows:  0 0 0 1 0 0 0
%        0 0 1 0 1 0 0
%        0 1 0 0 0 1 0
%        1 0 1 0 1 0 1
```
