function y = block_average(x, M)
%BLOCK_AVERAGE Downsample by averaging non-overlapping blocks of M samples.
if ~isscalar(M) || M < 1 || M ~= round(M)
    error('block_average:M', 'M must be a positive integer.');
end
nb = floor(numel(x) / M);
blocks = reshape(x(1:nb*M), M, nb);
y = mean(blocks, 1);
if iscolumn(x)
    y = y(:);
end
if nb == 0
    y = y([]);
end
end
