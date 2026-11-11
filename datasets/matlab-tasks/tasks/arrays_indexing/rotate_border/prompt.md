Write a MATLAB function

    B = rotate_border(A, k)

`A` is a matrix with at least 2 rows and at least 2 columns. Its **border**
is the ring of elements in the first and last rows and columns. Walk the
border clockwise starting at `A(1,1)`: along the first row to the right, down
the last column, along the last row to the left, and up the first column back
towards the start.

`B` equals `A` except that each border element has moved `k` steps along this
clockwise walk (wrapping around). A negative `k` moves elements
counterclockwise. Interior elements do not move. `k` is an integer and can be
larger than the number of border elements.

Examples:

    rotate_border([1 2 3; 4 5 6; 7 8 9], 1)
    % returns [4 1 2; 7 5 3; 8 9 6]

    rotate_border([1 2; 3 4], -1)
    % returns [2 4; 1 3]
