function [r, rho] = correlations(x, y)
%CORRELATIONS Pearson and Spearman correlation coefficients.
if numel(x) ~= numel(y)
    error('correlations:size', 'x and y must have the same length.');
end
x = x(:);
y = y(:);
keep = ~isnan(x) & ~isnan(y);
x = x(keep);
y = y(keep);
r = pearson(x, y);
rho = pearson(avg_ranks(x), avg_ranks(y));
end

function c = pearson(a, b)
if numel(a) < 2
    c = NaN;
    return
end
a = a - mean(a);
b = b - mean(b);
den = sqrt(sum(a.^2) * sum(b.^2));
if den == 0
    c = NaN;
else
    c = sum(a .* b) / den;
end
end

function rk = avg_ranks(v)
[s, idx] = sort(v);
n = numel(v);
rk = zeros(n, 1);
i = 1;
while i <= n
    j = i;
    while j < n && s(j + 1) == s(i)
        j = j + 1;
    end
    rk(idx(i:j)) = (i + j) / 2;
    i = j + 1;
end
end
