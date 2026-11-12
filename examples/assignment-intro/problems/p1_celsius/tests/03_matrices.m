% name: Columns, matrices and fractions
% points: 6
% hidden: true
f = c2f([-10; 0.5; 25]);
assert(isequal(size(f), [3 1]));
assert(all(abs(f - [14; 32.9; 77]) < 1e-9));
g = c2f([5 15; -5 -15]);
assert(isequal(size(g), [2 2]));
assert(all(all(abs(g - [41 59; 23 5]) < 1e-9)));
