Write a MATLAB function `[G, topples] = settle_sandpile(G)` that lets a pile
of sand grains settle on a rectangular grid.

`G` is an m-by-n matrix of nonnegative integers: the number of grains on each
square. A square holding 4 or more grains is unstable and *topples*: it loses
4 grains and each of its up to four orthogonal neighbours (up, down, left,
right) gains one grain. Grains pushed off the edge of the grid are lost.
Toppling continues until every square holds at most 3 grains.

The final grid does not depend on the order in which unstable squares are
toppled, and neither does the total number of topplings.

Return the stable grid `G` (same size, double) and `topples`, the total number
of single topplings performed (a square that topples three times counts 3).
Grids with many grains (thousands) must settle in reasonable time, so topple
all unstable squares of a sweep at once (or several times at once) rather than
one grain at a time.

Example:

```matlab
[G, t] = settle_sandpile([0 0 0; 0 4 0; 0 0 0])
% G = [0 1 0; 1 0 1; 0 1 0], t = 1
```
