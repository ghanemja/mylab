function [pks, locs] = find_local_peaks(x, minHeight, minDist)
%FIND_LOCAL_PEAKS Local maxima (plateau-aware) with height and spacing rules.
x = x(:)';
n = numel(x);
cand = zeros(1, 0);
i = 2;
while i <= n - 1
    if x(i) > x(i-1)
        j = i;
        while j < n && x(j+1) == x(i)
            j = j + 1;
        end
        if j < n && x(j+1) < x(i)
            cand(end+1) = i; %#ok<AGROW>
        end
        i = j + 1;
    else
        i = i + 1;
    end
end
cand = cand(x(cand) >= minHeight);
% Highest first; sortrows keeps ties in index order.
order = sortrows([-x(cand)' cand'], [1 2]);
kept = zeros(1, 0);
for c = order(:, 2)'
    if all(abs(c - kept) >= minDist)
        kept(end+1) = c; %#ok<AGROW>
    end
end
locs = sort(kept);
pks = x(locs);
end
