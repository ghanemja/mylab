To shrink an image, each non-overlapping `b-by-b` block of pixels is replaced
by its mean. Write a MATLAB function

    B = block_average(A, b)

where `A` is an `m-by-n` numeric matrix and `b` a positive integer. Both `m`
and `n` must be multiples of `b`; otherwise throw an error with identifier
`block_average:size`. `B` has size `(m/b)-by-(n/b)` and `B(i,j)` is the mean
of `A((i-1)*b+1 : i*b, (j-1)*b+1 : j*b)`. Return doubles.

Example:

    block_average([1 2 3 4; 5 6 7 8], 2)   % returns [3.5 5.5]
