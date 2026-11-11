x = gauss_solve([0  2  1; 3  -1  2; 1  1  4], [5; 7; 13]);
assert(isequal(size(x), [3 1]), 'column result');
assert(all(abs(x - [0.8750000000000003; 1.1250000000000002; 2.7499999999999996]) < 1e-12), 'zero leading pivot needs a swap');
X = gauss_solve([2  -1  0  0  3; 1  4  -2  1  0; 0  3  5  -1  2; 6  0  1  2  -1; 1  1  1  1  1], [1  0; 2  1; 3  0; 4  -1; 5  2]);
assert(isequal(size(X), [5 2]), 'several right-hand sides');
assert(max(abs(X(:) - reshape([-0.5901639344262292  -0.7905282331511839; 0.18032786885245897  0.02550091074681239; 0.9180327868852457  0.18032786885245902; 3.7049180327868845  2.0491803278688523; 0.7868852459016392  0.5355191256830599], [], 1))) < 1e-11, '5-by-5 system');
x3 = gauss_solve([1e-12  1; 1  1], [1; 2]);
assert(all(abs(x3 - [1.000000000001; 0.999999999999]) < 1e-12), 'tiny pivot handled by pivoting');
assert(abs(gauss_solve(4, 10) - 2.5) < 1e-15, '1-by-1 system');
threw = false;
try
    gauss_solve([1 2; 2 4], [1; 2]);
catch
    threw = true;
end
assert(threw, 'singular matrix must throw');
threw = false;
try
    gauss_solve([1 2 3; 4 5 6], [1; 2]);
catch
    threw = true;
end
assert(threw, 'non-square matrix must throw');
threw = false;
try
    gauss_solve(eye(3), [1; 2]);
catch
    threw = true;
end
assert(threw, 'wrong number of rows in B must throw');
