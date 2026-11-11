assert(isequal(in_exactly_one([1 2 2 3], [3; 4]), [1 2 4]), 'example 1');
c = in_exactly_one([5 5], 5);
assert(isempty(c) && isequal(size(c), [1 0]), 'example 2: 1x0');
assert(isequal(in_exactly_one([], [3 1 3]), [1 3]), 'one side empty');
c = in_exactly_one([], []);
assert(isequal(size(c), [1 0]), 'both empty');
assert(isequal(in_exactly_one([4; 2; -1], [2; 7]), [-1 4 7]), 'column inputs give a row');
assert(isequal(in_exactly_one([1 2; 3 4], [4 5 6]), [1 2 3 5 6]), 'matrix input');
assert(isequal(in_exactly_one([0.5 1.5], [1.5 2.5]), [0.5 2.5]), 'non-integers');
