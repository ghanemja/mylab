yq = lagrange_interp([0 1 2], [1 3 7], [3 -1]);
assert(all(abs(yq - [13 1]) < 1e-12), 'example');
yq = lagrange_interp([0 1 2.5 4], [1; -1; 2; 0.5], [-0.5 0.5 1.75 3 4.5]);
assert(isequal(size(yq), [1 5]) && all(abs(yq - [4.6625 -0.7249999999999999 0.23984375000000124 2.650000000000002 -3.274999999999997]) < 1e-12), 'cubic through uneven nodes');
assert(all(abs(lagrange_interp([0 1 2.5 4], [1 -1 2 0.5], [0 1 2.5 4]) - [1 -1 2 0.5]) < 1e-14), 'passes through the data');
Q = lagrange_interp([-1 1], [0 4], [0 1; 2 3]);
assert(isequal(size(Q), [2 2]) && all(all(abs(Q - [2 4; 6 8]) < 1e-14)), 'line, matrix of queries');
assert(isequal(lagrange_interp(2, 5, [0 10]), [5 5]), 'single point gives a constant');
xs = linspace(-1, 1, 6);
yq = lagrange_interp(xs, xs.^5 - 2*xs.^2, [0.3 -0.9]);
assert(all(abs(yq - ([0.3 -0.9].^5 - 2*[0.3 -0.9].^2)) < 1e-12), 'degree n-1 polynomial reproduced');
threw = false;
try
    lagrange_interp([0 1 1], [1 2 3], 0.5);
catch
    threw = true;
end
assert(threw, 'duplicate nodes must throw');
threw = false;
try
    lagrange_interp([0 1 2], [1 2], 0.5);
catch
    threw = true;
end
assert(threw, 'length mismatch must throw');
