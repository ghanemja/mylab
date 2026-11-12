% name: Streak at the start, at the end, and in a column
% points: 10
% hidden: true
assert(isequal(longest_streak([2 2 2 5 5 1]), 3));
assert(isequal(longest_streak([1 5 5 2 2 2 2]), 4));
assert(isequal(longest_streak([3; 3; 0; 0; 0; 3]), 3));
assert(isequal(longest_streak([0 1 0 1 0 1]), 1));
assert(isequal(longest_streak([-1 -1 1 1 1 -1 -1 -1 -1 -1]), 5));
assert(isequal(longest_streak([0.5 0.5 2.5]), 2));
