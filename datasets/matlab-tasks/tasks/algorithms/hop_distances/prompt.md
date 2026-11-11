Write a MATLAB function `d = hop_distances(A, src)`.

`A` is an n-by-n adjacency matrix of a **directed** graph: a nonzero value in
`A(i,j)` means there is an edge from node `i` to node `j` (values on the
diagonal can be ignored). `src` is a node index between 1 and n.

Return a 1-by-n row vector `d` where `d(k)` is the smallest number of edges on
a path from `src` to node `k`. `d(src)` is 0, and nodes that cannot be reached
from `src` get `Inf`.

Examples:

```matlab
A = [0 1 0 0
     0 0 1 0
     0 0 0 0
     1 0 0 0];
hop_distances(A, 1)   % returns [0 1 2 Inf]
hop_distances(A, 4)   % returns [1 2 3 0]
```
