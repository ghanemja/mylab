function r = integer_sqrt(n)
%INTEGER_SQRT floor(sqrt(n)) computed exactly for integers up to 2^53.
r = floor(sqrt(n));
while r * r > n
    r = r - 1;
end
while (r + 1) * (r + 1) <= n
    r = r + 1;
end
end
