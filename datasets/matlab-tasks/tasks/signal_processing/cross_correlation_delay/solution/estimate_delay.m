function [lag, peak] = estimate_delay(x, y, maxlag)
%ESTIMATE_DELAY Lag that maximizes the cross-correlation of x and y.
x = x(:);
y = y(:);
ex = sum(x.^2);
ey = sum(y.^2);
if ex == 0 || ey == 0
    error('estimate_delay:zero', 'Signals must not be all zeros.');
end
nx = numel(x);
ny = numel(y);
best = -Inf;
lag = 0;
for L = -maxlag:maxlag
    n = max(1, 1 - L):min(nx, ny - L);
    if isempty(n)
        c = 0;
    else
        c = sum(x(n) .* y(n + L));
    end
    better = c > best || (c == best && (abs(L) < abs(lag) || (abs(L) == abs(lag) && L < lag)));
    if better
        best = c;
        lag = L;
    end
end
peak = best / sqrt(ex * ey);
end
