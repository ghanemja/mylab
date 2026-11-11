[c, r] = fit_plane([0 0 1; 1 0 3; 0 1 4; 1 1 6]);
assert(isequal(size(c), [3 1]) && all(abs(c - [2; 3; 1]) < 1e-12), 'exact plane');
assert(r < 1e-12, 'zero residual');
[c, r] = fit_plane([0  0  101.2; 10  0  102.9; 20  0  104.1; 0  15  98.8; 10  15  100.6; 20  15  101.9; 5  30  96.9; 15  30  98.2; 25  25  100.1; 8  8  101.4]);
assert(all(abs(c - [0.1343284792166883; -0.1763301007203195; 101.52544357479182]) < 1e-10), 'survey plane coefficients');
assert(abs(r - 0.2379442486489036) < 1e-10, 'survey rms residual');
[c, r] = fit_plane([1 0 5; 0 1 5; 2 3 5]);
assert(all(abs(c - [0; 0; 5]) < 1e-12) && r < 1e-12, 'horizontal plane through three points');
[c, r] = fit_plane([0 0 0; 1 0 0; 0 1 0; 1 1 4]);
assert(all(abs(c - [2; 2; -1]) < 1e-12) && abs(r - 1) < 1e-12, 'one raised corner');
threw = false;
try
    fit_plane([0 0 1; 1 1 2]);
catch
    threw = true;
end
assert(threw, 'fewer than three points must throw');
