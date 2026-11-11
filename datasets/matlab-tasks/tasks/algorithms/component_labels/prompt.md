Write a MATLAB function `labels = component_labels(n, edges)`.

The undirected graph has nodes `1..n`. `edges` is an m-by-2 matrix; each row
`[a b]` connects nodes `a` and `b`. It may contain self-loops and repeated
edges, and it may be empty (`zeros(0,2)`).

Return a 1-by-n row vector `labels` giving each node the number of its
connected component. Components are numbered 1, 2, 3, ... in increasing order
of the smallest node index they contain, so node 1 always gets label 1.

Example:

```matlab
component_labels(6, [2 5; 4 1; 5 6])   % returns [1 2 3 1 2 2]
```

Node 3 has no edges, so it forms a component by itself.
