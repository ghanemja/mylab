function r = b28_sparse_tridiag_solve()
n = 200000; e = ones(n, 1);
A = spdiags([-e 2 * e -e], -1:1, n, n);
x = A \ e;
r = x(1);
end
