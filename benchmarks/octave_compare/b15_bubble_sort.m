function r = b15_bubble_sort()
x = mod((1:1500) * 7919, 1009);
n = numel(x);
for i = 1:n - 1
    for j = 1:n - i
        if x(j) > x(j + 1)
            t = x(j); x(j) = x(j + 1); x(j + 1) = t;
        end
    end
end
r = x(1);
end
