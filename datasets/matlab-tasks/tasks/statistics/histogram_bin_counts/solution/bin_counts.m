function [counts, centers] = bin_counts(x, edges)
%BIN_COUNTS Histogram counts for half-open bins (last bin closed).
edges = edges(:)';
if numel(edges) < 2 || any(diff(edges) <= 0)
    error('bin_counts:edges', 'edges must be strictly increasing with at least 2 elements.');
end
nb = numel(edges) - 1;
counts = zeros(1, nb);
x = x(:);
x = x(~isnan(x) & x >= edges(1) & x <= edges(end));
for i = 1:numel(x)
    if x(i) == edges(end)
        k = nb;
    else
        k = find(edges <= x(i), 1, 'last');
    end
    counts(k) = counts(k) + 1;
end
centers = (edges(1:end-1) + edges(2:end)) / 2;
end
