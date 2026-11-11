function T = running_balance(T)
%RUNNING_BALANCE Per-account cumulative sum, in row order.
n = height(T);
balance = zeros(n, 1);
if n > 0
    [~, ~, g] = unique(T.Account);
    current = zeros(max(g), 1);
    for r = 1:n
        current(g(r)) = current(g(r)) + T.Amount(r);
        balance(r) = current(g(r));
    end
end
T.Balance = balance;
end
