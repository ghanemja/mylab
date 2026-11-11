function yq = lagrange_interp(x, y, xq)
%LAGRANGE_INTERP Evaluate the interpolating polynomial in Lagrange form.
n = numel(x);
if numel(y) ~= n
    error('lagrange_interp:size', 'x and y must have the same length.');
end
if numel(unique(x)) < n
    error('lagrange_interp:duplicate', 'x values must be distinct.');
end
yq = zeros(size(xq));
for i = 1:n
    L = ones(size(xq));
    for j = [1:i-1, i+1:n]
        L = L .* (xq - x(j)) / (x(i) - x(j));
    end
    yq = yq + y(i) * L;
end
end
