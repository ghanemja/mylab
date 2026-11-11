function y = apply_repeatedly(f, x, n)
%APPLY_REPEATEDLY Apply F to X N times.
if n < 0 || n ~= round(n)
    error('apply_repeatedly:badCount', 'N must be a nonnegative integer.');
end
y = x;
for k = 1:n
    y = f(y);
end
end
