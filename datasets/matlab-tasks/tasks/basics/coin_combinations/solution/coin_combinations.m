function [ways, fewest] = coin_combinations(amount, coins)
%COIN_COMBINATIONS Number of coin multisets paying AMOUNT and the fewest coins.
w = zeros(1, amount + 1);   % w(k+1): ways to pay k
w(1) = 1;
f = inf(1, amount + 1);     % f(k+1): fewest coins to pay k
f(1) = 0;
for c = coins(:)'
    for k = c:amount
        w(k + 1) = w(k + 1) + w(k - c + 1);
        f(k + 1) = min(f(k + 1), f(k - c + 1) + 1);
    end
end
ways = w(amount + 1);
fewest = f(amount + 1);
end
