assert(fewest_tokens([1 3 4], 6) == 2, 'example 1');
assert(fewest_tokens([5 7], 3) == -1, 'example 2');
assert(fewest_tokens([2], 0) == 0, 'zero amount');
assert(fewest_tokens([7 3], 11) == -1, 'impossible combination');
assert(fewest_tokens([25 10 5 1], 63) == 6, 'ordinary coins');
assert(fewest_tokens([9 6 1], 12) == 2, 'greedy would give 4');
assert(fewest_tokens([3 7], 20) == 4, '7+7+3+3');
assert(fewest_tokens([1], 37) == 37, 'only ones');
assert(fewest_tokens([12 5 1], 16) == 4, '12+... vs 5+5+5+1');
