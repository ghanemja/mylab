Write a MATLAB function

    [B, count] = flood_fill(A, r, c, newValue)

that performs a "paint bucket" fill on the numeric matrix `A`. Starting at the
cell `A(r,c)`, find every cell that can be reached by moving up, down, left or
right (not diagonally) through cells whose value equals `A(r,c)`. `B` is `A`
with all those cells set to `newValue`, and `count` is the number of cells in
that region (including the start cell). If `newValue` already equals
`A(r,c)`, `B` equals `A` and `count` is still the size of the region.

If `(r, c)` is not a valid position in `A`, throw an error with identifier
`flood_fill:index`.

Example:

    A = [1 1 0 2
         0 1 0 2
         1 1 1 0
         0 0 1 2];
    [B, n] = flood_fill(A, 1, 1, 5)
    % B = [5 5 0 2; 0 5 0 2; 5 5 5 0; 0 0 5 2], n = 7
