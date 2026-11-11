function s = sum_of_multiples(factors, limit)
%SUM_OF_MULTIPLES Sum of distinct numbers below LIMIT divisible by any factor.
factors = factors(factors ~= 0);
n = 1:limit-1;
if isempty(n) || isempty(factors)
    s = 0;
    return
end
hit = false(size(n));
for f = factors(:)'
    hit = hit | mod(n, f) == 0;
end
s = sum(n(hit));
end
