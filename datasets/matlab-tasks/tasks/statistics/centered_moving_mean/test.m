y = centered_moving_mean([1 2 NaN 4 10], 3);
assert(isequal(size(y), [1 5]) && all(abs(y - [1.5 1.5 3 7 7]) < 1e-14), 'example');
y5 = centered_moving_mean([3.1; 3.4; NaN; 4.2; 5; 4.8; NaN; NaN; NaN; 6.1; 5.7; 5.2], 5);
assert(isequal(size(y5), [12 1]), 'column orientation kept');
assert(all(abs(y5 - [3.25; 3.5666666666666664; 3.925; 4.35; 4.666666666666667; 4.666666666666667; 4.9; 5.449999999999999; 5.9; 5.666666666666667; 5.666666666666667; 5.666666666666667]) < 1e-12), 'window 5 with gaps');
y3 = centered_moving_mean([3.1 3.4 NaN 4.2 5 4.8 NaN NaN NaN 6.1 5.7 5.2], 3);
assert(isequal(isnan(y3), [0 0 0 0 0 0 0 1 0 0 0 0] == 1), 'all-NaN window gives NaN');
ok = ~isnan(y3);
expected = [3.25 3.25 3.8 4.6 4.666666666666667 4.9 4.8 NaN 6.1 5.9 5.666666666666667 5.45];
assert(all(abs(y3(ok) - expected(ok)) < 1e-12), 'window 3 values');
assert(isequal(centered_moving_mean([4 5 6], 1), [4 5 6]), 'window 1 returns the data');
yl = centered_moving_mean([2 4 6 8], 9);
assert(all(abs(yl - 5) < 1e-14), 'window longer than the data');
threw = false;
try
    centered_moving_mean([1 2 3], 4);
catch
    threw = true;
end
assert(threw, 'even window must throw');
threw = false;
try
    centered_moving_mean([1 2 3], 0);
catch
    threw = true;
end
assert(threw, 'zero window must throw');
