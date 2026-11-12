% name: Empty vector and single value
% points: 5
assert(isequal(longest_streak([]), 0), 'An empty vector has no streak: the answer is 0');
assert(isequal(longest_streak(5), 1), 'A single value is a streak of length 1');
