function r = cfh_fact(n)
% Recursive factorial.
if n <= 1
    r = 1;
else
    r = n * cfh_fact(n - 1);
end
end
