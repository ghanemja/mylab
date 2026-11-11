[c, v] = newton_interp([0 1 3], [1 2 10], 2);
assert(isequal(size(c), [1 3]) && all(abs(c - [1 1 1]) < 1e-14) && abs(v - 5) < 1e-14, 'example');
[c, v] = newton_interp([1; 2; 4; 7; 8], [2; 3.5; 1; 6; 4.5], [0 3 5.5 9]);
assert(isequal(size(c), [1 5]) && all(abs(c - [2 1.5 -0.9166666666666666 0.25 -0.06845238095238095]) < 1e-13), 'divided differences, column input');
assert(isequal(size(v), [1 4]) && all(abs(v - [-7.166666666666664 2.1190476190476417 2.644531250000039 -5.66666666666651]) < 1e-10), 'evaluation including extrapolation');
[c, v] = newton_interp([0 2 5 9], [18 21.4 24.1 22.7], [3.5; 7]);
assert(all(abs(c - [18 1.6999999999999993 -0.15999999999999967 -0.0020634920634921223]) < 1e-13) && isequal(size(v), [2 1]) && all(abs(v - [23.12625000000001; 24.1555555555556]) < 1e-12), 'temperature log, column query');
[c1, v1] = newton_interp([3 1 2], [9 1 4], [0 5]);
assert(all(abs(c1 - [9 4 1]) < 1e-14) && all(abs(v1 - [0 25]) < 1e-12), 'unsorted nodes reproduce q^2');
[c0, v0] = newton_interp(4, 7, [1 2 3]);
assert(c0 == 7 && isequal(v0, [7 7 7]), 'single node');
threw = false;
try
    newton_interp([1 2 1], [0 1 2], 0);
catch
    threw = true;
end
assert(threw, 'duplicate nodes must throw');
