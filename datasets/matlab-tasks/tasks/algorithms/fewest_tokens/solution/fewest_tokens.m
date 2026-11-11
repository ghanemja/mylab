function k = fewest_tokens(values, amount)
%FEWEST_TOKENS Minimum number of tokens summing to amount (unbounded).
best = inf(1, amount + 1);   % best(a+1) = tokens needed for amount a
best(1) = 0;
for a = 1:amount
    for v = values(:)'
        if v <= a && best(a - v + 1) + 1 < best(a + 1)
            best(a + 1) = best(a - v + 1) + 1;
        end
    end
end
k = best(amount + 1);
if isinf(k)
    k = -1;
end
end
