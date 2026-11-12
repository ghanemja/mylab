[a, b] = deal(1, 2);
[c, d] = deal(7);
[e{1:3}] = deal(0);
s = struct('x', {1, 2});
[s.x] = deal(9);
sx = [s.x];
[f, g, h] = deal('a', [1 2], {3});
t1 = 0; try, [m, n] = deal(1, 2, 3); catch, t1 = 1; end
z = deal(5);
try, [q{1:2}] = deal('p', 'q'); catch, end
w = cell(1, 2);
[w{:}] = deal(1);
