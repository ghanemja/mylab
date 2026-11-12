function n = longest_streak(v)
% LONGEST_STREAK  Length of the longest group of equal neighbours in v.

n = 0;
current = 0;
for k = 1:numel(v)
    if k > 1 && v(k) == v(k - 1)
        current = current + 1;
    else
        current = 1;
    end
    if current > n
        n = current;
    end
end

end
