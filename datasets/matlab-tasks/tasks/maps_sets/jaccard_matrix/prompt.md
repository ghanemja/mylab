Write a MATLAB function `J = jaccard_matrix(sets)`.

`sets` is a cell array of n numeric vectors; each vector describes a set (its
repeated values count once, and the order does not matter). Return the
n-by-n matrix `J` with

    J(i,j) = |S_i ∩ S_j| / |S_i ∪ S_j|

the Jaccard similarity of sets i and j. When both sets are empty, define
`J(i,j) = 1`. `J` is symmetric with ones on the diagonal. For an empty cell
array, return `zeros(0,0)`.

Example:

```matlab
J = jaccard_matrix({[1 2 3], [2 3 4 4], []})
% J = [1    0.5  0
%      0.5  1    0
%      0    0    1]
```
