assert(isequal(base_digits(10, 2), [1 0 1 0]), 'binary');
assert(isequal(base_digits(255, 16), [15 15]), 'hex');
assert(isequal(base_digits(0, 7), 0), 'zero');
assert(isequal(base_digits(7, 10), 7), 'single decimal digit');
assert(isequal(base_digits(100, 3), [1 0 2 0 1]), 'base 3');
d = base_digits(1000000, 10);
assert(isequal(size(d), [1 7]) && isequal(d, [1 0 0 0 0 0 0]), 'row of seven digits');
assert(isequal(base_digits(36, 36), [1 0]), 'base equal to n');
assert(isa(base_digits(5, 2), 'double'), 'double output');
