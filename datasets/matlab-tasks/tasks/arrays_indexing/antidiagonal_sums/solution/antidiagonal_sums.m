function s = antidiagonal_sums(A)
%ANTIDIAGONAL_SUMS Sum of each anti-diagonal i+j-1 = k, as a row vector.
[m, n] = size(A);
if m == 0 || n == 0
    s = zeros(1, 0);
    return
end
s = zeros(1, m + n - 1);
for i = 1:m
    for j = 1:n
        s(i + j - 1) = s(i + j - 1) + A(i, j);
    end
end
end
