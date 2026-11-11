A 3-D array `A` of size `m-by-n-by-p` stores `p` pages (2-D slices
`A(:,:,k)`). Write a MATLAB function

    [k, sums] = heaviest_page(A)

that returns

- `sums`: a `1-by-p` **row vector** whose `k`-th element is the sum of all
  elements of page `k`;
- `k`: the index of the page with the largest sum (the smallest such index if
  several pages tie).

A 2-D matrix counts as an array with a single page. Assume `A` is nonempty.

Example:

    A = cat(3, [1 2; 3 4], [5 5; 5 5], [0 0; 0 20]);
    [k, s] = heaviest_page(A)   % k = 2, s = [10 20 20]
