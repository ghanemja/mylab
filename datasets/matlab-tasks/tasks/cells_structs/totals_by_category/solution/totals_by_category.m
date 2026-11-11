function t = totals_by_category(expenses)
%TOTALS_BY_CATEGORY Sum of amounts per category, as dynamic struct fields.
t = struct();
for k = 1:numel(expenses)
    name = expenses(k).category;
    if isfield(t, name)
        t.(name) = t.(name) + expenses(k).amount;
    else
        t.(name) = expenses(k).amount;
    end
end
end
