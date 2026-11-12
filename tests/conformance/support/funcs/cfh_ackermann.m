function r = cfh_ackermann(m, n)
% Ackermann function, deep mutual recursion on small inputs.
if m == 0
    r = n + 1;
elseif n == 0
    r = cfh_ackermann(m - 1, 1);
else
    r = cfh_ackermann(m - 1, cfh_ackermann(m, n - 1));
end
end
