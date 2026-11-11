function yq = pchip_eval(x, y, xq)
%PCHIP_EVAL Shape-preserving piecewise cubic Hermite interpolation.
x = x(:);
y = y(:);
n = numel(x);
if n < 2 || any(diff(x) <= 0)
    error('pchip_eval:x', 'x must be strictly increasing with at least 2 points.');
end
h = diff(x);
del = diff(y) ./ h;
d = zeros(n, 1);
if n == 2
    d(:) = del(1);
else
    for k = 2:n-1
        if sign(del(k-1)) * sign(del(k)) > 0
            w1 = 2*h(k) + h(k-1);
            w2 = h(k) + 2*h(k-1);
            d(k) = (w1 + w2) / (w1/del(k-1) + w2/del(k));
        end
    end
    d(1) = end_slope(h(1), h(2), del(1), del(2));
    d(n) = end_slope(h(n-1), h(n-2), del(n-1), del(n-2));
end
yq = NaN(size(xq));
for j = 1:numel(xq)
    q = xq(j);
    if isnan(q) || q < x(1) || q > x(n)
        continue
    end
    k = find(x <= q, 1, 'last');
    if k == n
        k = n - 1;
    end
    t = (q - x(k)) / h(k);
    h00 = 2*t^3 - 3*t^2 + 1;
    h10 = t^3 - 2*t^2 + t;
    h01 = -2*t^3 + 3*t^2;
    h11 = t^3 - t^2;
    yq(j) = h00*y(k) + h10*h(k)*d(k) + h01*y(k+1) + h11*h(k)*d(k+1);
end
end

function d = end_slope(h1, h2, del1, del2)
d = ((2*h1 + h2)*del1 - h1*del2) / (h1 + h2);
if sign(d) ~= sign(del1)
    d = 0;
elseif sign(del1) ~= sign(del2) && abs(d) > abs(3*del1)
    d = 3*del1;
end
end
