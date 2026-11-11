assert(isequal(block_average([1 3 5 7 9 11 13], 3), [3 9]), 'example');
yc = block_average((1:10)', 2);
assert(isequal(size(yc), [5 1]) && isequal(yc, [1.5; 3.5; 5.5; 7.5; 9.5]), 'column input');
x = sin(2*pi*(0:99)/100);
y = block_average(x, 10);
assert(isequal(size(y), [1 10]) && all(abs(y - mean(reshape(x, 10, 10))) < 1e-15), '100 Hz to 10 Hz');
assert(isequal(block_average([4 8 15], 1), [4 8 15]), 'factor 1');
assert(isequal(block_average([2 4 6 8], 4), 5), 'single block');
assert(isempty(block_average([1 2], 3)), 'too few samples');
threw = false;
try
    block_average([1 2 3], 1.5);
catch
    threw = true;
end
assert(threw, 'non-integer factor must throw');
threw = false;
try
    block_average([1 2 3], 0);
catch
    threw = true;
end
assert(threw, 'zero factor must throw');
