function [nums, texts, others] = split_by_class(c)
%SPLIT_BY_CLASS Separate numeric scalars, char arrays and everything else.
nums = zeros(1, 0);
texts = cell(1, 0);
others = cell(1, 0);
for k = 1:numel(c)
    x = c{k};
    if isnumeric(x) && isscalar(x)
        nums(end + 1) = double(x); %#ok<AGROW>
    elseif ischar(x)
        texts{end + 1} = x; %#ok<AGROW>
    else
        others{end + 1} = x; %#ok<AGROW>
    end
end
end
