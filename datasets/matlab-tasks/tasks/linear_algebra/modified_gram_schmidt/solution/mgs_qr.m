function [Q, R] = mgs_qr(A)
%MGS_QR Thin QR factorization by modified Gram-Schmidt.
[m, n] = size(A);
if m < n
    error('mgs_qr:shape', 'A must have at least as many rows as columns.');
end
Q = A;
R = zeros(n);
for k = 1:n
    R(k, k) = norm(Q(:, k));
    if R(k, k) == 0 || R(k, k) <= 1e-10 * norm(A(:, k))
        error('mgs_qr:rank', 'Columns are linearly dependent.');
    end
    Q(:, k) = Q(:, k) / R(k, k);
    for j = k+1:n
        R(k, j) = Q(:, k)' * Q(:, j);
        Q(:, j) = Q(:, j) - R(k, j) * Q(:, k);
    end
end
end
