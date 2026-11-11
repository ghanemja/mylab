A = [0 1 0 0; 0 0 1 0; 0 0 0 0; 1 0 0 0];
assert(isequal(hop_distances(A, 1), [0 1 2 Inf]), 'example 1');
assert(isequal(hop_distances(A, 4), [1 2 3 0]), 'example 2');
assert(isequal(hop_distances(A, 3), [Inf Inf 0 Inf]), 'sink node');
assert(isequal(hop_distances(0, 1), 0), 'single node');
% shortcut edge must be preferred over the long chain
B = zeros(6);
B(1,2) = 1; B(2,3) = 1; B(3,4) = 1; B(4,5) = 1; B(5,6) = 1; B(1,5) = 1;
assert(isequal(hop_distances(B, 1), [0 1 2 3 1 2]), 'shortcut');
% undirected cycle given as symmetric logical matrix
n = 7;
C = false(n);
for k = 1:n
    C(k, mod(k, n) + 1) = true;
    C(mod(k, n) + 1, k) = true;
end
assert(isequal(hop_distances(C, 1), [0 1 2 3 3 2 1]), 'cycle');
d = hop_distances(B, 6);
assert(isequal(size(d), [1 6]) && all(isinf(d(1:5))), 'row output, unreachable');
