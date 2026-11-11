assert(isequal(greedy_unit_fractions(2, 3), [2 6]), '2/3');
assert(isequal(greedy_unit_fractions(4, 13), [4 18 468]), '4/13');
assert(isequal(greedy_unit_fractions(1, 7), 7), 'already a unit fraction');
assert(isequal(greedy_unit_fractions(3, 4), [2 4]), '3/4');
assert(isequal(greedy_unit_fractions(6, 14), [3 11 231]), 'unreduced input 6/14');
d = greedy_unit_fractions(5, 6);
assert(isequal(size(d), [1 2]) && isequal(d, [2 3]), 'row output');
d = greedy_unit_fractions(7, 15);
assert(isequal(d, [3 8 120]), 'three terms');
