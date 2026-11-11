assert(isequal(first_pair_with_sum([4 1 3 2 5], 5), [1 2]), 'example');
assert(isequal(first_pair_with_sum([1 6 3 3 2], 6), [3 4]), 'same value twice');
assert(isempty(first_pair_with_sum([2 2], 3)), 'no pair');
assert(isempty(first_pair_with_sum([3], 6)), 'single element cannot pair with itself');
assert(isempty(first_pair_with_sum([], 0)), 'empty');
assert(isequal(first_pair_with_sum([5 9 1 4], 10), [2 3]), 'smallest j wins over smallest i');
assert(isequal(first_pair_with_sum([-2; 7; 0; 2], 0), [1 4]), 'column and negatives');
p = first_pair_with_sum([1 2 3], 5);
assert(isequal(size(p), [1 2]) && isequal(p, [2 3]), 'row output');
