[c, m] = bin_counts([2.3 4.0 5.5 7.9 8.0 10.0 12.0], 0:2:10);
assert(isequal(c, [0 1 2 1 2]) && isequal(m, [1 3 5 7 9]), 'example');
[c, m] = bin_counts([2.3; 4; 5.5; 7.9; 8; 3.3; 12; 0.4; NaN; 6.1; 9.99; 10; -1; 4; 15.2], [0; 2; 4; 6; 8; 10]);
assert(isequal(size(c), [1 5]) && isequal(size(m), [1 5]), 'row outputs for column inputs');
assert(isequal(c, [1 2 3 2 3]), 'wind speed counts with NaN and out-of-range values');
assert(isequal(m, [1 3 5 7 9]), 'centers');
[c2, m2] = bin_counts([-3 -0.5 -0.5 0 0.5 3 3 2.9 3.1], [-3 -0.5 0.5 3]);
assert(isequal(c2, [1 3 4]) && all(abs(m2 - [-1.75 0 1.75]) < 1e-15), 'uneven bins and edge values');
assert(isequal(bin_counts([1 2; 3 4], [0 5]), 4), 'matrix input, single bin');
assert(isequal(bin_counts([], [0 1 2]), [0 0]), 'empty data');
assert(strcmp(class(bin_counts([1 2], [0 1 2])), 'double'), 'class double');
threw = false;
try
    bin_counts([1 2 3], [0 2 2 4]);
catch
    threw = true;
end
assert(threw, 'non-increasing edges must throw');
threw = false;
try
    bin_counts([1 2 3], 5);
catch
    threw = true;
end
assert(threw, 'a single edge must throw');
