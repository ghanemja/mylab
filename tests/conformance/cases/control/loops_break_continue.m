a1 = [];
for k = 1:10
    if mod(k, 2) == 0
        continue;
    end
    if k > 7
        break;
    end
    a1(end+1) = k;
end
kend = k;
a2 = 0;
n = 0;
while true
    n = n + 1;
    if n > 5
        break;
    end
    a2 = a2 + n;
end
a3 = [];
for i = 1:3
    for j = 1:3
        if j == 2
            continue;
        end
        if i == 2
            break;
        end
        a3(end+1, :) = [i j];
    end
end
a4 = 0;
w = 10;
while w > 0
    w = w - 3;
    a4 = a4 + 1;
end
a5 = 0;
for q = 5:-1:1
    a5 = a5 * 10 + q;
end
a6 = [];
m = 0;
while m < 3
    m = m + 1;
    if m == 2
        continue;
    end
    a6(end+1) = m;
end
a7 = 0;
for k2 = 1:5
    k2 = k2 * 2;
    a7 = a7 + k2;
end
k2final = k2;
a8 = 0;
while []
    a8 = 1;
end
a9 = 0;
while [1 1 0]
    a9 = 1;
    break;
end
