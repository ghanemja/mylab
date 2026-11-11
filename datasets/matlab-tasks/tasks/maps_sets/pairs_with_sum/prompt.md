Write a MATLAB function `P = pairs_with_sum(v, target)`.

`v` is a vector of integers (it may contain repeated values) and `target` is
an integer. Return every pair of positions `[i j]` with `i < j` and
`v(i) + v(j) == target`, as a k-by-2 matrix whose rows are sorted by `i`, then
by `j`. If there are no such pairs, return `zeros(0,2)`.

Aim for a solution that does not test all n² pairs when only a few match, for
example by remembering the positions of each value in a `containers.Map`.

Example:

```matlab
pairs_with_sum([3 1 4 1 2], 5)
% returns [1 5; 2 3; 3 4]
```
