o = simple_regression([1 2 3 4], [3 5 7 9]);
assert(abs(o.slope - 2) < 1e-14 && abs(o.intercept - 1) < 1e-14 && abs(o.r2 - 1) < 1e-14, 'exact line');
assert(o.se_slope < 1e-7 && o.se_intercept < 1e-7 && all(abs(o.residuals) < 1e-13), 'exact line errors');
o = simple_regression([0.5 1 1.5 2 2.5 3 3.5 4], [1.9 3.1 3.8 5.2 5.9 7.3 7.8 9.4]);
assert(abs(o.slope - 2.0761904761904764) < 1e-12 && abs(o.intercept - 0.8785714285714281) < 1e-12, 'dose-response fit');
assert(abs(o.r2 - 0.9930002716312502) < 1e-12, 'r squared');
assert(abs(o.se_slope - 0.0711635294332059) < 1e-12, 'slope standard error');
assert(abs(o.se_intercept - 0.17967910423548453) < 1e-12, 'intercept standard error');
assert(isequal(size(o.residuals), [8 1]) && all(abs(o.residuals - [-0.016666666666666385; 0.1452380952380956; -0.19285714285714306; 0.16904761904761934; -0.16904761904761845; 0.19285714285714217; -0.3452380952380958; 0.21666666666666679]) < 1e-12), 'residual column');
assert(abs(sum(o.residuals)) < 1e-12, 'residuals sum to zero');
threw = false;
try
    simple_regression([1 2], [3 4]);
catch
    threw = true;
end
assert(threw, 'two points must throw');
threw = false;
try
    simple_regression([2 2 2], [1 2 3]);
catch
    threw = true;
end
assert(threw, 'constant x must throw');
threw = false;
try
    simple_regression([1 2 3], [1 2 3 4]);
catch
    threw = true;
end
assert(threw, 'length mismatch must throw');
