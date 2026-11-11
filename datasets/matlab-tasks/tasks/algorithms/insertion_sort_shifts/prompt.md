Write a MATLAB function `[s, shifts] = insertion_sort_shifts(v)` that sorts the
numeric vector `v` into ascending order using insertion sort, implemented with
your own loops (do not call `sort`, `sortrows` or similar built-ins).

Insertion sort walks through the vector from the second element onwards. The
current element is held aside, and every element to its left that is
**strictly greater** than it is moved one position to the right; the held
element is then dropped into the gap. Each single move of one element one
position to the right counts as one *shift*. Equal elements are never moved
past each other.

Outputs:

- `s` — the sorted vector, with the same orientation (row or column) and size
  as `v`.
- `shifts` — the total number of shifts performed (a double scalar).

An empty input returns an empty `s` of the same size and `shifts = 0`.

Examples:

- `[s, k] = insertion_sort_shifts([3 1 2])` gives `s = [1 2 3]`, `k = 2`.
- `[s, k] = insertion_sort_shifts([2; 2; 1])` gives `s = [1; 2; 2]`, `k = 2`.
