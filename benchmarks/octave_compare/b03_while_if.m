function r = b03_while_if()
n = 0; c = 0;
while n < 1000000
    n = n + 1;
    if mod(n, 3) == 0
        c = c + 1;
    end
end
r = c;
end
