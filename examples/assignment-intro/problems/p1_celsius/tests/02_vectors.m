% name: A row of temperatures
% points: 6
f = c2f([0 10 20 30]);
assert(isequal(size(f), [1 4]), 'The output must have the same size as the input');
assert(all(abs(f - [32 50 68 86]) < 1e-9), 'c2f([0 10 20 30]) should be [32 50 68 86]');
