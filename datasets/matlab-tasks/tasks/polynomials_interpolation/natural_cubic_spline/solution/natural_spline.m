function yq = natural_spline(x, y, xq)
%NATURAL_SPLINE Evaluate the natural cubic spline through (x, y).
x = x(:);
y = y(:);
n = numel(x);
if n < 3 || any(diff(x) <= 0)
    error('natural_spline:x', 'x must be strictly increasing with at least 3 points.');
end
h = diff(x);
% Second derivatives M(2..n-1); M(1) = M(n) = 0.
m = n - 2;
A = zeros(m);
rhs = zeros(m, 1);
for i = 1:m
    k = i + 1;
    A(i, i) = 2 * (h(k-1) + h(k));
    if i > 1
        A(i, i-1) = h(k-1);
    end
    if i < m
        A(i, i+1) = h(k);
    end
    rhs(i) = 6 * ((y(k+1) - y(k)) / h(k) - (y(k) - y(k-1)) / h(k-1));
end
M = [0; A \ rhs; 0];
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
    a = x(k + 1) - q;
    b = q - x(k);
    yq(j) = (M(k)*a^3 + M(k+1)*b^3) / (6*h(k)) ...
        + (y(k)/h(k) - M(k)*h(k)/6) * a + (y(k+1)/h(k) - M(k+1)*h(k)/6) * b;
end
end
