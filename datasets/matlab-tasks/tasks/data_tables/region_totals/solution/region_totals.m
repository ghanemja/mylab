function S = region_totals(T)
%REGION_TOTALS Sum and count of sales per region.
[regions, ~, g] = unique(T.Region);
regions = regions(:);
g = g(:);
n = numel(regions);
Total = accumarray(g, T.Sales, [n 1]);
Count = accumarray(g, 1, [n 1]);
S = table(regions, Total, Count, 'VariableNames', {'Region', 'Total', 'Count'});
end
