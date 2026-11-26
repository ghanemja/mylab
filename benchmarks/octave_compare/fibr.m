function r = fibr(n)
if n < 2
    r = n;
else
    r = fibr(n - 1) + fibr(n - 2);
end
end
