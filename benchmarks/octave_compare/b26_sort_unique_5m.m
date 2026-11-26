function r = b26_sort_unique_5m()
x = mod((1:5e6) * 7919, 1000003);
y = sort(x);
u = unique(mod(x, 1000));
r = y(1) + numel(u);
end
