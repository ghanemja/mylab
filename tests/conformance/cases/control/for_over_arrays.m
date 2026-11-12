c1 = {};
for col = [1 2 3; 4 5 6]
    c1{end+1} = col;
end
c2 = {};
for item = {1, 'a', [2 3]}
    c2{end+1} = class(item);
end
c3 = 0;
for v = [1; 2; 3]
    c3 = c3 + numel(v);
end
c4 = [];
for v = 'hey'
    c4(end+1) = double(v);
end
c5 = {};
for s = struct('a', {1, 2})
    c5{end+1} = s.a;
end
c6 = {};
for z = reshape(1:8, 2, 2, 2)
    c6{end+1} = z';
end
c7 = 0;
for v = 7
    c7 = c7 + v;
end
c8 = {};
for e = {1; 2}
    c8{end+1} = size(e);
end
c9 = [];
for x = int8([1 2 3])
    c9 = [c9 class(x) ' '];
end
c10 = 0;
for t = [true false true]
    if t
        c10 = c10 + 1;
    end
end
