[n, d] = reduce_fraction(6, 8);
assert(n == 3 && d == 4, 'simple');
[n, d] = reduce_fraction(-6, -8);
assert(n == 3 && d == 4, 'both negative');
[n, d] = reduce_fraction(6, -8);
assert(n == -3 && d == 4, 'negative denominator');
[n, d] = reduce_fraction(0, -5);
assert(n == 0 && d == 1, 'zero numerator');
[n, d] = reduce_fraction(7, 1);
assert(n == 7 && d == 1, 'already whole');
[n, d] = reduce_fraction(-91, 1001);
assert(n == -1 && d == 11, 'larger common factor');
ok = false;
try
    reduce_fraction(3, 0);
catch err
    ok = strcmp(err.identifier, 'reduce_fraction:zeroDenominator');
end
assert(ok, 'zero denominator error');
