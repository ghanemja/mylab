function M = pad_to_matrix(c, fill)
%PAD_TO_MATRIX Stack vectors of different lengths as rows, padding with FILL.
if nargin < 2
    fill = NaN;
end
if isempty(c)
    M = zeros(0, 0);
    return
end
lens = cellfun(@numel, c);
M = repmat(fill, numel(c), max(lens));
for k = 1:numel(c)
    M(k, 1:lens(k)) = reshape(c{k}, 1, []);
end
end
