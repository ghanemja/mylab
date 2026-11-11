function B = rotate_border(A, k)
%ROTATE_BORDER Rotate the outer ring of A clockwise by K positions.
[m, n] = size(A);
r = [ones(1, n), 2:m, m * ones(1, n - 1), m-1:-1:2];
c = [1:n, n * ones(1, m - 1), n-1:-1:1, ones(1, m - 2)];
idx = sub2ind([m n], r, c);
B = A;
B(idx) = circshift(A(idx), [0 k]);
end
