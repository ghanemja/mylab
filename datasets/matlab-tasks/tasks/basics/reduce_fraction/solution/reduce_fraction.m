function [n, d] = reduce_fraction(num, den)
%REDUCE_FRACTION Lowest terms with a positive denominator.
if den == 0
    error('reduce_fraction:zeroDenominator', 'Denominator must not be zero.');
end
if num == 0
    n = 0;
    d = 1;
    return
end
g = gcd(num, den);
n = num / g;
d = den / g;
if d < 0
    n = -n;
    d = -d;
end
end
