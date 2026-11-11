function yq = pw_linear(x, y, xq)
%PW_LINEAR Linear interpolation in a table, NaN outside the range.
x = x(:);
y = y(:);
n = numel(x);
yq = NaN(size(xq));
for i = 1:numel(xq)
    q = xq(i);
    if isnan(q) || q < x(1) || q > x(n)
        continue
    end
    k = find(x <= q, 1, 'last');
    if k == n
        yq(i) = y(n);
    else
        t = (q - x(k)) / (x(k+1) - x(k));
        yq(i) = y(k) + t * (y(k+1) - y(k));
    end
end
end
