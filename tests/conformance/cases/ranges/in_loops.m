acc = [];
for k = 1:0
    acc(end+1) = k;
end
sacc = size(acc);
acc2 = [];
for k = 3:-1:1
    acc2(end+1) = k;
end
acc3 = [];
for k = 0:0.5:1.5
    acc3(end+1) = k * 2;
end
acc4 = 0;
for k = [1 2; 3 4]
    acc4 = acc4 + k(1) * k(2);
end
for kk = 1:3
end
acc5 = 0;
for tmpk = []
    acc5 = acc5 + 1;
end
for tmpk = zeros(0, 3)
end
acc7 = 0;
for tmpk = zeros(3, 0)
    acc7 = acc7 + 1;
end
acc8 = '';
for ch = 'abc'
    acc8 = [ch acc8];
end
for z = 5
end
for e1 = 1:0
end
try, se1 = size(e1); catch, end
