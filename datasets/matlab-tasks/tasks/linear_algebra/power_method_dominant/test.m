A = [4  -1  0  0; -1  4  -1  0; 0  -1  4  -1; 0  0  -1  3];
[lam, v, it] = power_iteration(A, [1; 0; 0; 0], 1e-14, 1000);
assert(abs(lam - 5.532088886237956) < 1e-10, 'dominant eigenvalue');
assert(isequal(size(v), [4 1]) && abs(norm(v) - 1) < 1e-12, 'unit column vector');
assert(all(abs(v - [-0.42852507312435983; 0.6565385020081386; -0.5773502691896255; 0.22801342888377915]) < 1e-5), 'dominant eigenvector with sign convention');
assert(it > 1 && it < 1000, 'iteration count is reported');
[lam2, v2] = power_iteration(diag([-5 2 1]), [1 1 1], 1e-13, 500);
assert(abs(lam2 + 5) < 1e-10 && all(abs(v2 - [1; 0; 0]) < 1e-5), 'negative dominant eigenvalue');
[lam3, v3, it3] = power_iteration([2 0; 0 1], [0; 1], 1e-12, 50);
assert(abs(lam3 - 1) < 1e-12 && it3 == 1, 'start vector orthogonal to the dominant one');
[lam4, ~, it4] = power_iteration([2 1; 1 3], [1; 0], 1e-15, 3);
assert(it4 == 3, 'stops after maxit');
B = [2 1; 1 3]; v = [1; 0];
for k = 1:3
    v = B*v / norm(B*v);
end
assert(abs(lam4 - v'*B*v) < 1e-12, 'returns the estimate after maxit steps');
