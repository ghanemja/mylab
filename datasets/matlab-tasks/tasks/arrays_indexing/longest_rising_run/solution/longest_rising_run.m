function [start, len] = longest_rising_run(x)
%LONGEST_RISING_RUN First longest strictly increasing contiguous run.
if isempty(x)
    start = 0;
    len = 0;
    return
end
start = 1;
len = 1;
curStart = 1;
for k = 2:numel(x)
    if x(k) <= x(k - 1)
        curStart = k;
    end
    if k - curStart + 1 > len
        len = k - curStart + 1;
        start = curStart;
    end
end
end
