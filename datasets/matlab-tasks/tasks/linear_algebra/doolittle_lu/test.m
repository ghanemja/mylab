[L, U] = doolittle_lu([2  1; 6  8]);
assert(isequal(L, [1  0; 3  1]) && isequal(U, [2  1; 0  5]), 'example');
A = [4  3  -2  1; 8  4  -1  5; -4  5  6  2; 12  7  3  9];
[L, U] = doolittle_lu(A);
assert(max(max(abs(L - [1  0  0  0; 2  1  0  0; -1  -4  1  0; 3  1  0.375  1]))) < 1e-13, '4-by-4 L');
assert(max(max(abs(U - [4  3  -2  1; 0  -2  3  3; 0  0  16  15; 0  0  0  -2.625]))) < 1e-13, '4-by-4 U');
assert(isequal(diag(L), ones(4, 1)) && isequal(triu(L, 1), zeros(4)), 'L is unit lower triangular');
assert(isequal(tril(U, -1), zeros(4)), 'U is upper triangular');
assert(max(max(abs(L*U - A))) < 1e-12, 'L*U reproduces A');
[L1, U1] = doolittle_lu(7);
assert(L1 == 1 && U1 == 7, '1-by-1');
[Ls, Us] = doolittle_lu([1 2; 3 6]);
assert(abs(Us(2, 2)) < 1e-14 && isequal(Ls, [1 0; 3 1]), 'singular matrix with a zero last pivot');
threw = false;
try
    doolittle_lu([0 1; 1 0]);
catch
    threw = true;
end
assert(threw, 'zero pivot must throw');
threw = false;
try
    doolittle_lu([1 2 3; 4 5 6]);
catch
    threw = true;
end
assert(threw, 'non-square must throw');
