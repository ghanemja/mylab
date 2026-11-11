function X = gauss_solve(A, B)
%GAUSS_SOLVE Solve A*X = B by Gaussian elimination with partial pivoting.
[n, nc] = size(A);
if n ~= nc
    error('gauss_solve:square', 'A must be square.');
end
if size(B, 1) ~= n
    error('gauss_solve:size', 'B must have as many rows as A.');
end
tol = n * eps * max(abs(A(:)));
M = [A, B];
for k = 1:n
    [~, p] = max(abs(M(k:n, k)));
    p = p + k - 1;
    if abs(M(p, k)) <= tol
        error('gauss_solve:singular', 'Matrix is singular to working precision.');
    end
    if p ~= k
        M([k p], :) = M([p k], :);
    end
    for i = k+1:n
        factor = M(i, k) / M(k, k);
        M(i, k:end) = M(i, k:end) - factor * M(k, k:end);
    end
end
U = M(:, 1:n);
C = M(:, n+1:end);
X = zeros(n, size(B, 2));
for i = n:-1:1
    X(i, :) = (C(i, :) - U(i, i+1:n) * X(i+1:n, :)) / U(i, i);
end
end
