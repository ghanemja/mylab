function out = simple_regression(x, y)
%SIMPLE_REGRESSION Ordinary least-squares line with standard errors.
x = x(:);
y = y(:);
n = numel(x);
if numel(y) ~= n || n < 3
    error('simple_regression:size', 'Need equal-length vectors with at least 3 points.');
end
mx = mean(x);
my = mean(y);
Sxx = sum((x - mx).^2);
if Sxx == 0
    error('simple_regression:degenerate', 'All x values are equal.');
end
out.slope = sum((x - mx) .* (y - my)) / Sxx;
out.intercept = my - out.slope * mx;
out.residuals = y - (out.intercept + out.slope * x);
SSE = sum(out.residuals.^2);
SST = sum((y - my).^2);
out.r2 = 1 - SSE / SST;
s2 = SSE / (n - 2);
out.se_slope = sqrt(s2 / Sxx);
out.se_intercept = sqrt(s2 * (1/n + mx^2 / Sxx));
end
