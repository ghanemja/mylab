function out = flatten_nested(c)
%FLATTEN_NESTED Depth-first list of all non-cell values in nested cells.
out = cell(1, 0);
for k = 1:numel(c)
    x = c{k};
    if iscell(x)
        out = [out, flatten_nested(x)]; %#ok<AGROW>
    else
        out{end + 1} = x; %#ok<AGROW>
    end
end
end
