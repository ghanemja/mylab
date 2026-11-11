function [coef, yq] = newton_interp(x, y, xq)
%NEWTON_INTERP Divided-difference coefficients and evaluation of the Newton form.
x = x(:)';
n = numel(x);
if numel(unique(x)) < n
    error('newton_interp:duplicate', 'Nodes must be distinct.');
end
coef = y(:)';
for j = 2:n
    for i = n:-1:j
        coef(i) = (coef(i) - coef(i-1)) / (x(i) - x(i-j+1));
    end
end
yq = coef(n) * ones(size(xq));
for k = n-1:-1:1
    yq = yq .* (xq - x(k)) + coef(k);
end
end
