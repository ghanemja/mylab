function [k, sums] = heaviest_page(A)
%HEAVIEST_PAGE Index of the page with the largest sum, and all page sums.
p = size(A, 3);
sums = reshape(sum(reshape(A, [], p), 1), 1, p);
[~, k] = max(sums);
end
