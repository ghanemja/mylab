x = thomas_solve([1 1], [4 4 4], [1 1], [5 6 5]);
assert(isequal(size(x), [3 1]) && all(abs(x - 1) < 1e-14), 'example');
x = thomas_solve([1 -2 0.5 3], [10; 9; -8; 7; 12], [2 1 -1 0.5], [1 2 3 4 5]);
assert(all(abs(x - [0.045060999606454155; 0.2746950019677292; -0.5173160173160174; 0.58913813459268; 0.2693821330184967]) < 1e-13), 'general tridiagonal system');
dx = 1/7; k = 2;
a = -k/dx^2*ones(5, 1); c = a; b = 2*k/dx^2*ones(6, 1);
d = [5; 5; 10; 10; 5; 5]; d(1) = d(1) + k/dx^2*20; d(end) = d(end) + k/dx^2*80;
T = thomas_solve(a, b, c, d);
assert(all(abs(T - [28.775510204081627; 37.49999999999999; 46.1734693877551; 54.744897959183675; 63.214285714285715; 71.63265306122449]) < 1e-10), 'heated rod temperatures');
assert(abs(thomas_solve([], 5, [], 20) - 4) < 1e-15, 'single equation');
x2 = thomas_solve(3, [2 5], -1, [1 13]);
assert(all(abs(x2 - [18/13; 23/13]) < 1e-14), 'two equations');
threw = false;
try
    thomas_solve([1 1 1], [4 4 4], [1 1], [5 6 5]);
catch
    threw = true;
end
assert(threw, 'inconsistent lengths must throw');
