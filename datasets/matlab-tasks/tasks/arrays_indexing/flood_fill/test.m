A = [1 1 0 2; 0 1 0 2; 1 1 1 0; 0 0 1 2];
[B, n] = flood_fill(A, 1, 1, 5);
assert(isequal(B, [5 5 0 2; 0 5 0 2; 5 5 5 0; 0 0 5 2]) && n == 7, 'example');
[B, n] = flood_fill(A, 1, 4, 9);
assert(isequal(B, [1 1 0 9; 0 1 0 9; 1 1 1 0; 0 0 1 2]) && n == 2, 'separate region not filled');
[B, n] = flood_fill(A, 2, 1, 7);
assert(n == 1 && B(2, 1) == 7 && B(4, 1) == 0, 'no diagonal moves');
[B, n] = flood_fill(A, 4, 3, 1);
assert(isequal(B, A) && n == 7, 'same value');
[B, n] = flood_fill(zeros(3, 4), 3, 4, -1);
assert(isequal(B, -ones(3, 4)) && n == 12, 'whole matrix');
S = [0 0 0 0 0; 1 1 1 1 0; 0 0 0 1 0; 0 1 1 1 0; 0 0 0 0 0];
[B, n] = flood_fill(S, 3, 1, 8);
assert(n == 17 && B(1, 1) == 8 && B(3, 3) == 8 && B(5, 5) == 8 && B(2, 1) == 1, 'winding path');
ok = false;
try
    flood_fill(A, 5, 1, 0);
catch err
    ok = strcmp(err.identifier, 'flood_fill:index');
end
assert(ok, 'bad index');
