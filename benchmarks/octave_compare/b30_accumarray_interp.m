function r = b30_accumarray_interp()
idx = mod((1:2e6)' * 7919, 1000) + 1;
a = accumarray(idx, 1);
v = interp1(1:1000, a', linspace(1, 1000, 2e6));
r = sum(v);
end
