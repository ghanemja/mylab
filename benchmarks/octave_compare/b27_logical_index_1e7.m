function r = b27_logical_index_1e7()
x = mod((1:1e7) * 7919, 1009);
y = x(x > 500);
k = find(x == 7);
r = numel(y) + numel(k) + sum(cumsum(y) > 0);
end
