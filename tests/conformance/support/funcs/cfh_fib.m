function r = cfh_fib(n)
% Doubly recursive Fibonacci number.
if n < 2
    r = n;
    return;
end
r = cfh_fib(n - 1) + cfh_fib(n - 2);
end
