function r = b02_array_fill()
x = zeros(1, 1000000);
for k = 1:1000000
    x(k) = k * 2;
end
r = x(end);
end
