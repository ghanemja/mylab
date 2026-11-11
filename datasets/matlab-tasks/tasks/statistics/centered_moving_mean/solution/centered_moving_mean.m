function y = centered_moving_mean(x, k)
%CENTERED_MOVING_MEAN Centered moving average, shrinking at ends, NaN-aware.
if ~isscalar(k) || k < 1 || k ~= round(k) || mod(k, 2) == 0
    error('centered_moving_mean:k', 'k must be a positive odd integer.');
end
h = (k - 1) / 2;
n = numel(x);
y = zeros(size(x));
for i = 1:n
    w = x(max(1, i - h):min(n, i + h));
    w = w(~isnan(w));
    if isempty(w)
        y(i) = NaN;
    else
        y(i) = sum(w) / numel(w);
    end
end
end
