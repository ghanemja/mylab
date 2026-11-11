assert(sum_of_multiples([3 5], 10) == 23, 'example 1');
assert(sum_of_multiples(7, 7) == 0, 'limit equal to factor');
assert(sum_of_multiples([3 5], 1) == 0, 'nothing below 1');
assert(sum_of_multiples([4 6], 20) == 4+6+8+12+16+18, 'overlapping multiples counted once');
assert(sum_of_multiples([0 2], 7) == 12, 'zero factor ignored');
assert(sum_of_multiples([], 10) == 0, 'no factors');
