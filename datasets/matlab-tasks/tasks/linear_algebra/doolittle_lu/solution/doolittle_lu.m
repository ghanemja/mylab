function [L, U] = doolittle_lu(A)
%DOOLITTLE_LU LU factorization without pivoting (unit lower L).
[n, m] = size(A);
if n ~= m
    error('doolittle_lu:square', 'A must be square.');
end
tol = 1e-12 * norm(A, inf);
L = eye(n);
U = zeros(n);
for i = 1:n
    for j = i:n
        U(i, j) = A(i, j) - L(i, 1:i-1) * U(1:i-1, j);
    end
    if i < n && abs(U(i, i)) <= tol
        error('doolittle_lu:pivot', 'Zero pivot at step %d.', i);
    end
    for j = i+1:n
        L(j, i) = (A(j, i) - L(j, 1:i-1) * U(1:i-1, i)) / U(i, i);
    end
end
end
