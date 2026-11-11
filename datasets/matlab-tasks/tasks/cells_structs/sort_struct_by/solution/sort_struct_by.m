function s = sort_struct_by(s, field, direction)
%SORT_STRUCT_BY Stable ascending/descending sort of a struct array by a field.
if nargin < 3
    direction = 'ascend';
end
if ~isfield(s, field)
    error('sort_struct_by:noField', 'No field "%s".', field);
end
if isempty(s)
    return
end
vals = {s.(field)};
if ischar(vals{1})
    [~, ~, r] = unique(vals);
else
    [~, ~, r] = unique([vals{:}]);
end
r = reshape(r, 1, []);
if strcmp(direction, 'descend')
    r = -r;
end
[~, order] = sort(r);
s = s(order);
end
