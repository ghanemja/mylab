function yq = local_linear_smooth(x, y, xq, bw)
%LOCAL_LINEAR_SMOOTH Local linear regression with a Gaussian kernel.
if numel(x) ~= numel(y)
    error('local_linear_smooth:size', 'x and y must have the same length.');
end
if ~(bw > 0)
    error('local_linear_smooth:bw', 'Bandwidth must be positive.');
end
x = x(:);
y = y(:);
allSame = all(x == x(1));
yq = zeros(size(xq));
for i = 1:numel(xq)
    q = xq(i);
    w = exp(-0.5 * ((x - q) / bw).^2);
    W = sum(w);
    xbar = sum(w .* x) / W;
    ybar = sum(w .* y) / W;
    if allSame
        yq(i) = ybar;
    else
        dx = x - xbar;
        b = sum(w .* dx .* (y - ybar)) / sum(w .* dx.^2);
        yq(i) = ybar + b * (q - xbar);
    end
end
end
