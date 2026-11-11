function I = simpson_composite(f, a, b, n)
%SIMPSON_COMPOSITE Composite Simpson's 1/3 rule with n subintervals.
if ~isscalar(n) || n <= 0 || n ~= round(n) || mod(n, 2) ~= 0
    error('simpson_composite:n', 'n must be a positive even integer.');
end
h = (b - a) / n;
x = a + (0:n) * h;
y = f(x);
I = h/3 * (y(1) + y(end) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)));
end
