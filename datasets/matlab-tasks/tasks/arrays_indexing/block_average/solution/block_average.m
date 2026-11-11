function B = block_average(A, b)
%BLOCK_AVERAGE Mean of each non-overlapping b-by-b block.
[m, n] = size(A);
if mod(m, b) ~= 0 || mod(n, b) ~= 0
    error('block_average:size', 'Matrix size must be a multiple of the block size.');
end
T = reshape(double(A), b, m / b, b, n / b);
T = sum(sum(T, 1), 3);
B = reshape(T, m / b, n / b) / b^2;
end
