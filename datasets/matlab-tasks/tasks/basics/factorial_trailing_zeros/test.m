assert(factorial_trailing_zeros(0) == 0, 'zero');
assert(factorial_trailing_zeros(4) == 0, '4!');
assert(factorial_trailing_zeros(5) == 1, '5!');
assert(factorial_trailing_zeros(10) == 2, '10!');
assert(factorial_trailing_zeros(25) == 6, '25! has extra factor of 5');
assert(factorial_trailing_zeros(124) == 28, '124!');
assert(factorial_trailing_zeros(125) == 31, '125!');
assert(factorial_trailing_zeros(1e9) == 249999998, 'large n');
