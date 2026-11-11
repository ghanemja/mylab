function P = monthly_pivot(T)
%MONTHLY_PIVOT Category-by-month totals as a wide table.
[cats, ~, gc] = unique(T.Category);
[months, ~, gm] = unique(T.Month);
cats = cats(:);
S = accumarray([gc(:), gm(:)], T.Amount, [numel(cats), numel(months)]);
P = table(cats, 'VariableNames', {'Category'});
for k = 1:numel(months)
    P.(sprintf('M%d', months(k))) = S(:, k);
end
end
