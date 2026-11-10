% functions_demo.m - scripts can define local functions at the end
n = 10;
fprintf('fib(%d) = %d\n', n, fib(n));

[mn, mx, avg] = stats([4 8 15 16 23 42]);
fprintf('min %g, max %g, mean %g\n', mn, mx, avg);

function r = fib(n)
    if n < 2
        r = n;
    else
        r = fib(n - 1) + fib(n - 2);
    end
end

function [lo, hi, m] = stats(v)
    lo = min(v);
    hi = max(v);
    m = mean(v);
end
