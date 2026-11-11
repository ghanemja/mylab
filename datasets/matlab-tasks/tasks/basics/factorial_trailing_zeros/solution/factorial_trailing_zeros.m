function z = factorial_trailing_zeros(n)
%FACTORIAL_TRAILING_ZEROS Number of trailing decimal zeros of n!.
z = 0;
p = 5;
while p <= n
    z = z + floor(n / p);
    p = p * 5;
end
end
