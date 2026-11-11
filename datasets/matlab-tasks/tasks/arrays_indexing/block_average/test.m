assert(isequal(block_average([1 2 3 4; 5 6 7 8], 2), [3.5 5.5]), 'example');
A = reshape(1:16, 4, 4);
assert(isequal(block_average(A, 2), [3.5 11.5; 5.5 13.5]), '4x4 in 2x2 blocks');
assert(isequal(block_average(A, 4), 8.5), 'one block');
assert(isequal(block_average(A, 1), A), 'block size 1');
B = block_average(ones(6, 9), 3);
assert(isequal(size(B), [2 3]) && all(B(:) == 1), 'rectangular result size');
B = block_average([1 0 0 0 0 0; 0 0 0 0 0 9], 2);
assert(isequal(B, [0.25 0 2.25]), 'uneven contents');
ok = false;
try
    block_average(ones(4, 6), 4);
catch err
    ok = strcmp(err.identifier, 'block_average:size');
end
assert(ok, 'size error');
