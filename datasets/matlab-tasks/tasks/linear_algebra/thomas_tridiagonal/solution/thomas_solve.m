function x = thomas_solve(a, b, c, d)
%THOMAS_SOLVE Solve a tridiagonal system with the Thomas algorithm.
n = numel(b);
if numel(d) ~= n || numel(a) ~= n - 1 || numel(c) ~= n - 1
    error('thomas_solve:size', 'Inconsistent diagonal lengths.');
end
a = a(:); b = b(:); c = c(:); d = d(:);
cp = zeros(n, 1);
dp = zeros(n, 1);
if n > 1
    cp(1) = c(1) / b(1);
end
dp(1) = d(1) / b(1);
for i = 2:n
    m = b(i) - a(i-1) * cp(i-1);
    if i < n
        cp(i) = c(i) / m;
    end
    dp(i) = (d(i) - a(i-1) * dp(i-1)) / m;
end
x = zeros(n, 1);
x(n) = dp(n);
for i = n-1:-1:1
    x(i) = dp(i) - cp(i) * x(i+1);
end
end
