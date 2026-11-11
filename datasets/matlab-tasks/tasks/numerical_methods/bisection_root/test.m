[r, n] = bisect_root(@(x) cos(x) - x, 0, 1, 1e-10);
assert(abs(r - 0.7390851332151607) <= 1e-10, 'cos(x) = x root');
assert(n == 33, 'step count for cos(x) = x');
g = @(T) 0.8*5.67e-8*(T.^4 - 293^4) + 10*(T - 293) - 1500;
[T, nT] = bisect_root(g, 293, 1000, 1e-6);
assert(abs(T - 380.92474320276347) <= 1e-6, 'plate temperature');
assert(nT == 29, 'step count for plate temperature');
[r1, n1] = bisect_root(@(x) x - 1, 0, 2, 1e-8);
assert(r1 == 1 && n1 == 1, 'exact hit at the first midpoint');
[r2, n2] = bisect_root(@(x) x.^2 - 4, 2, 5, 1e-6);
assert(r2 == 2 && n2 == 0, 'root at left endpoint');
[r3, n3] = bisect_root(@(x) x - 3, -1, 3, 1e-6);
assert(r3 == 3 && n3 == 0, 'root at right endpoint');
[r4, n4] = bisect_root(@(x) 2 - x, 0, 10, 1e-3);
assert(abs(r4 - 2) <= 1e-3 && n4 > 0, 'decreasing function');
threw = false;
try
    bisect_root(@(x) x.^2 + 1, -1, 1, 1e-6);
catch
    threw = true;
end
assert(threw, 'no sign change must throw');
