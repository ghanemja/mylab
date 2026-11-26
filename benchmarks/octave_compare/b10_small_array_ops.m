function r = b10_small_array_ops()
t = 0;
for k = 1:100000
    v = [k, k + 1, k + 2];
    t = t + sum(v) * max(v) - norm(v);
end
r = t;
end
