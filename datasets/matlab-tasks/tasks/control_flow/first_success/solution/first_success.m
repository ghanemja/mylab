function [y, idx] = first_success(fns, x)
%FIRST_SUCCESS Result of the first handle in FNS that runs without error.
for idx = 1:numel(fns)
    try
        y = fns{idx}(x);
        return
    catch
        % try the next one
    end
end
error('first_success:allFailed', 'All %d attempts failed.', numel(fns));
end
