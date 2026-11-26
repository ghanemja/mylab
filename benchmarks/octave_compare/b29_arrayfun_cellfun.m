function r = b29_arrayfun_cellfun()
y = arrayfun(@(x) x^2 + 1, 1:100000);
c = num2cell(1:100000);
z = cellfun(@(x) x * 2, c);
r = sum(y) + sum(z);
end
