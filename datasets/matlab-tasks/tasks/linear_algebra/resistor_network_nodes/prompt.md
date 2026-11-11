A DC circuit consists of resistors connecting nodes `0, 1, ..., nnodes`.
Node 0 is ground (0 V). An ideal voltage source holds node 1 at `vs` volts
relative to ground.

Write a MATLAB function `[v, Isrc] = node_voltages(edges, nnodes, vs)` where
`edges` is a `k`-by-3 matrix; row `[i j R]` is a resistor of `R` ohms between
nodes `i` and `j`. Several resistors may join the same pair of nodes.

- `v` is an `nnodes`-by-1 column vector of the voltages at nodes `1..nnodes`
  (so `v(1) = vs`).
- `Isrc` is the total current (amperes) that flows from node 1 into the
  resistors connected to it, i.e. the current the source delivers.

Use nodal analysis: assemble the conductance matrix (conductance `1/R` per
resistor), fix the known voltages of nodes 0 and 1, and solve the linear
system for the remaining nodes. You may assume every node is connected to
ground or node 1 through some path, so the system is nonsingular.

Example: a divider `edges = [1 2 1000; 2 0 1000]` with `nnodes = 2`,
`vs = 10` gives `v = [10; 5]` and `Isrc = 0.005`.
